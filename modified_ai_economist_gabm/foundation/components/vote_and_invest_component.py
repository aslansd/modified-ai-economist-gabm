# Modified by Aslan Satary Dizaji, Copyright (c) 2024.

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy
import numpy as np

from modified_ai_economist_gabm.foundation.base.base_component import BaseComponent, component_registry
from modified_ai_economist_gabm.foundation.components.redistribution import PeriodicBracketTax

epsilon = 1e-19


@component_registry.add
class AgentVotesAgentInvestsResources(BaseComponent):
    """
    Allows mobile agents to vote for and to invest on three resources (wood, stone, and iron).

    Args:
        disable_taxes (bool): Whether to disable any tax collection, effectively
            enforcing that tax rates are always 0. Useful for removing taxes without
            changing the observation space. Default is False (taxes enabled).
        tax_model (str): Which tax model to use for setting taxes.
            "model_wrapper" (default) uses the actions of the planner agent;
            "saez" uses an adaptation of the theoretical optimal taxation formula
            derived in https://www.nber.org/papers/w7628.
            "us-federal-single-filer-2018-scaled" uses US federal tax rates from 2018;
            "fixed-bracket-rates" uses the rates supplied in fixed_bracket_rates.
        period (int): Length of a tax period in environment timesteps. Taxes are
            updated at the start of each period and collected/redistributed at the
            end of each period. Must be > 0. Default is 100 timesteps.
        rate_min (float): Minimum tax rate within a bracket. Must be >= 0 (default).
        rate_max (float): Maximum tax rate within a bracket. Must be <= 1 (default).
        rate_disc (float): (Only applies for "model_wrapper") the interval separating
            discrete tax rates that the planner can select. Default of 0.05 means,
            for example, the planner can select among [0.0, 0.05, 0.10, ... 1.0].
            Must be > 0 and < 1.
        n_brackets (int): How many tax brackets to use. Must be >=2. Default is 5.
        top_bracket_cutoff (float): The income at the left end of the last tax
            bracket. Must be >= 10. Default is 100 coin.
        usd_scaling (float): Scale by which to divide the US Federal bracket cutoffs
            when using bracket_spacing = "us-federal". Must be > 0. Default is 1000.
        bracket_spacing (str): How bracket cutoffs should be spaced.
            "us-federal" (default) uses scaled cutoffs from the 2018 US Federal
                taxes, with scaling set by usd_scaling (ignores n_brackets and
                top_bracket_cutoff);
            "linear" linearly spaces the n_bracket cutoffs between 0 and
                top_bracket_cutoff;
            "log" is similar to "linear" but with logarithmic spacing.
        fixed_bracket_rates (list): Required if tax_model=="fixed-bracket-rates". A
            list of fixed marginal rates to use for each bracket. Length must be
            equal to the number of brackets (7 for "us-federal" spacing, n_brackets
            otherwise).
        pareto_weight_type (str): Type of Pareto weights to use when computing tax
            rates using the Saez formula. "inverse_income" (default) uses 1/z;
            "uniform" uses 1.
        saez_fixed_elas (float, optional): If supplied, this value will be used as
            the elasticity estimate when computing tax rates using the Saez formula.
            If not given (default), elasticity will be estimated empirically.
        tax_annealing_schedule (list, optional): A length-2 list of
            [tax_annealing_warmup, tax_annealing_slope] describing the tax annealing
            schedule. See annealed_tax_mask function for details. Default behavior is
            no tax annealing.     
    """

    name = "AgentVotesAgentInvestsResources"
    component_type = "VoteInvest"
    required_entities = ["Wood", "Stone", "Iron", "Coin", "VoteInvest"]
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]
    
    def __init__(
        self,
        *base_component_args,

        disable_taxes=False,
        tax_model='model_wrapper',
        period=100,
        rate_min=0.0,
        rate_max=1.0,
        rate_disc=0.05,
        n_brackets=7,
        top_bracket_cutoff=100,
        usd_scaling=1000.0,
        bracket_spacing='us-federal',
        fixed_bracket_rates=None,
        pareto_weight_type='inverse_income',
        saez_fixed_elas=None,
        tax_annealing_schedule=None,
        
        **base_component_kwargs
    ):
        
        super().__init__(*base_component_args, **base_component_kwargs)
        
        ### From redistribution module

        self.disable_taxes=disable_taxes
        self.tax_model=tax_model
        self.period=period
        self.rate_min=rate_min
        self.rate_max=rate_max
        self.rate_disc=rate_disc
        self.n_brackets=n_brackets
        self.top_bracket_cutoff=top_bracket_cutoff
        self.usd_scaling=usd_scaling
        self.bracket_spacing=bracket_spacing
        self.fixed_bracket_rates=fixed_bracket_rates
        self.pareto_weight_type=pareto_weight_type
        self.saez_fixed_elas=saez_fixed_elas
        self.tax_annealing_schedule=tax_annealing_schedule
        
        self.periodic_bracket_tax = PeriodicBracketTax(
            world=self.world,
            episode_length=self.episode_length,
            disable_taxes=self.disable_taxes,
            tax_model=self.tax_model,
            period=self.period,
            rate_min=self.rate_min,
            rate_max=self.rate_max,
            rate_disc=self.rate_disc,
            n_brackets=self.n_brackets,
            top_bracket_cutoff=self.top_bracket_cutoff,
            usd_scaling=self.usd_scaling,
            bracket_spacing=self.bracket_spacing,
            fixed_bracket_rates=self.fixed_bracket_rates,
            pareto_weight_type=self.pareto_weight_type,
            saez_fixed_elas=self.saez_fixed_elas,
            tax_annealing_schedule=self.tax_annealing_schedule,
        )
        
        ###
        
        self.votesinvests = []

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.
        
        Add six actions (voting for and investing on three resources using one of six possibilities) for mobile agents.
        
        Also, if using the "model_wrapper" tax model and taxes are enabled, the planner's action space includes an action 
        subspace for each of the tax brackets. Each such action space has as many actions as there are discretized tax rates.
        """
        
        # This component adds six actions that mobile agents can take when vote for and invest on three different resources.
        if agent_cls_name == "BasicMobileAgent":
            return 6
        
        # Also, the planner takes actions through tax scheduling.
        if agent_cls_name == "BasicPlanner":
            if self.tax_model == "model_wrapper" and not self.disable_taxes:
                # For every bracket, the planner can select one of the discretized tax rates.
                return [
                    ("TaxIndexBracket_{:03d}".format(int(r)), self.periodic_bracket_tax.n_disc_rates)
                    for r in self.periodic_bracket_tax.bracket_cutoffs
                ]
    
    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        For mobile agents and the planner, add a state field indicating the vote counting method.
        """
        
        if agent_cls_name not in self.agent_subclasses:
            return {}
        else:
            return {"vote_count_method": "Not required!"}
        
        raise NotImplementedError

    def component_step(self):
        """
        See base_component.py for detailed description.
        
        On the first day of each tax period, update taxes. On the last day, enact them.
        
        Also, rank three resources (wood, stone, and iron) and using the collected taxes to invest on them.
        """
        
        # 1. On the first day of a new tax period: set up the taxes for this period.
        if self.periodic_bracket_tax.tax_cycle_pos == 1:
            if self.tax_model == "model_wrapper":
                self.periodic_bracket_tax.set_new_period_rates_model()
    
            if self.tax_model == "saez":
                self.periodic_bracket_tax.compute_and_set_new_period_rates_from_saez_formula()
    
            self.periodic_bracket_tax._curr_rates_obs = np.array(self.periodic_bracket_tax.curr_marginal_rates)
    
        # 2. On the last day of the tax period: get $-taxes and update agent endowments.
        if self.periodic_bracket_tax.tax_cycle_pos >= self.period:
            self.periodic_bracket_tax.enact_taxes()
            self.periodic_bracket_tax.tax_cycle_pos = 0
    
        else:
            self.periodic_bracket_tax.taxes.append([])
    
        # Increment timestep.
        self.periodic_bracket_tax.tax_cycle_pos += 1
        
        voteinvest = []
        investments = np.zeros(3, float)

        matrix_order = np.zeros((6, 3), int)
        resource_array = ["Wood", "Stone", "Iron"]

        matrix_order[0, :] = [3, 2, 1]
        matrix_order[1, :] = [3, 1, 2]
        matrix_order[2, :] = [2, 3, 1]
        matrix_order[3, :] = [2, 1, 3]
        matrix_order[4, :] = [1, 3, 2]
        matrix_order[5, :] = [1, 2, 3]
        
        for agent in self.world.get_random_order_agents():
            action = agent.get_component_action(self.name)
                
            income = self.periodic_bracket_tax.last_income[agent.idx]
            
            investment = self.periodic_bracket_tax.taxes_due(income)

            action_flag = False

            # NO-OP!
            if action == 0 or action is None:
                action_flag = True
                
                pass

            # Vote for and invest on wood, stone, and iron!
            for i in range(6):
                if action_flag == True:
                    break

                if action == i + 1:
                    investments = investments + np.array([matrix_order[i, 0] * investment / 6, matrix_order[i, 1] * investment / 6, matrix_order[i, 2] * investment / 6])
                
                    voteinvest.append(
                        {
                            "voterinvestor": agent.idx, 
                            "agents": ["Agent", "Agent"],
                            "vote": [resource_array[matrix_order[i, 2] - 1], resource_array[matrix_order[i, 1] - 1], resource_array[matrix_order[i, 0] - 1]],
                            "invest": investments,
                        }
                    )
                
                    agent.state["endogenous"]["VoteInvest"] = investments

                    action_flag = True
            
            if action_flag == False:
                # raise ValueError
                pass
        
        self.votesinvests.append(voteinvest)
    
        return investments
    
    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Agents observe where in the tax period cycle they are, information about the last period's incomes, and the current 
        marginal tax rates, including the marginal rate that will apply to their next unit of income.

        Also, the planner observes the same type of information, but for all the agents. It also sees, for each agent, their 
        marginal tax rate and reported income from the previous tax period.
        
        Moreover, agents observe their choice of vote and investment. The planner does not observe any vote and investment actions.
        """
        
        is_tax_day = float(self.periodic_bracket_tax.tax_cycle_pos >= self.period)
        is_first_day = float(self.periodic_bracket_tax.tax_cycle_pos == 1)
        tax_phase = self.periodic_bracket_tax.tax_cycle_pos / self.period

        obs = dict()
        
        obs[self.world.planner.idx] = dict(
            is_tax_day=is_tax_day,
            is_first_day=is_first_day,
            tax_phase=tax_phase,
            last_incomes=self.periodic_bracket_tax._last_income_obs_sorted,
            curr_rates=self.periodic_bracket_tax._curr_rates_obs,
        )
        
        for agent in self.world.agents:           
            i = agent.idx
            k = str(i)

            curr_marginal_rate = self.periodic_bracket_tax.marginal_rate(
                agent.total_endowment("Coin") - self.periodic_bracket_tax.last_coin[i]
            )

            obs[k] = dict(
                is_tax_day=is_tax_day,
                is_first_day=is_first_day,
                tax_phase=tax_phase,
                last_incomes=self.periodic_bracket_tax._last_income_obs_sorted,
                curr_rates=self.periodic_bracket_tax._curr_rates_obs,
                marginal_rate=curr_marginal_rate,
                vote_invest_action=agent.state["endogenous"]["VoteInvest"],
            )

            obs["p" + k] = dict(
                last_income=self.periodic_bracket_tax._last_income_obs[i],
                last_marginal_rate=self.periodic_bracket_tax.last_marginal_rate[i],
                curr_marginal_rate=curr_marginal_rate,
            )

        return obs
    
    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.
        
        Tax scheduling masks only apply to the planner if tax_model == "model_wrapper" and taxes are enabled. 
        All tax actions are masked (so, only NO-OPs can be sampled) on all timesteps except when self.tax_cycle_pos==1 
        (meaning a new tax period is starting). When self.tax_cycle_pos==1, tax actions are masked in order to enforce 
        any tax annealing.
        
        Moreover, agents can always vote.
        """
        
        if (
            completions != self.periodic_bracket_tax._last_completions
            and self.tax_annealing_schedule is not None
        ):
            self.periodic_bracket_tax._last_completions = int(completions)
            self.periodic_bracket_tax._annealed_rate_max = self.periodic_bracket_tax.annealed_tax_limit(
                completions,
                self.periodic_bracket_tax._annealing_warmup,
                self.periodic_bracket_tax._annealing_slope,
                self.rate_max,
            )

        if self.disable_taxes:
            return {}

        if self.tax_model == "model_wrapper":
            # No annealing. Generate masks using default method.
            if self.tax_annealing_schedule is None:
                if self.periodic_bracket_tax._planner_masks is None:
                    masks = super().generate_masks(completions=completions)
                    self.periodic_bracket_tax._planner_masks = dict(
                        new_taxes=deepcopy(masks[self.world.planner.idx]),
                        zeros={
                            k: np.zeros_like(v)
                            for k, v in masks[self.world.planner.idx].items()
                        },
                    )

                # No need to recompute. Use the cached masks.
                masks = dict()
                if self.periodic_bracket_tax.tax_cycle_pos != 1 or self.disable_taxes:
                    # Apply zero masks for any timestep where taxes are not going to be updated.
                    masks[self.world.planner.idx] = self.periodic_bracket_tax._planner_masks["zeros"]
                else:
                    masks[self.world.planner.idx] = self.periodic_bracket_tax._planner_masks["new_taxes"]

            # Doing annealing.
            else:
                # Figure out what the masks should be this episode.
                if self.periodic_bracket_tax._planner_masks is None:
                    planner_masks = {
                        k: self.periodic_bracket_tax.annealed_tax_mask(
                            completions,
                            self.periodic_bracket_tax._annealing_warmup,
                            self.periodic_bracket_tax._annealing_slope,
                            tax_values,
                        )
                        for k, tax_values in self.periodic_bracket_tax._planner_tax_val_dict.items()
                    }
                    self.periodic_bracket_tax._planner_masks = dict(
                        new_taxes=deepcopy(planner_masks),
                        zeros={k: np.zeros_like(v) for k, v in planner_masks.items()},
                    )

                # No need to recompute. Use the cached masks.
                masks = dict()
                if self.periodic_bracket_tax.tax_cycle_pos != 1 or self.disable_taxes:
                    # Apply zero masks for any timestep where taxes are not going to be updated.
                    masks[self.world.planner.idx] = self.periodic_bracket_tax._planner_masks["zeros"]
                else:
                    masks[self.world.planner.idx] = self.periodic_bracket_tax._planner_masks["new_taxes"]

        # We are not using a learned planner. Generate masks by the default method.
        else:
            masks = super().generate_masks(completions=completions)
            
        for agent in self.world.agents:
            masks[agent.idx] = np.array([1, 1, 1, 1, 1, 1])

        return masks
    
    def get_metrics(self):
        """
        Metrics that capture what happened through this component.

        Return metrics related to bracket rates, bracket occupancy, and tax collection.
        """
        
        out = dict()

        n_observed_incomes = np.maximum(1, np.sum(list(self.periodic_bracket_tax._occupancy.values())))
        for c in self.periodic_bracket_tax.bracket_cutoffs:
            k = "{:03d}".format(int(c))
            out["avg_bracket_rate/{}".format(k)] = np.mean(self.periodic_bracket_tax._schedules[k])
            out["bracket_occupancy/{}".format(k)] = (
                self.periodic_bracket_tax._occupancy[k] / n_observed_incomes
            )

        if not self.disable_taxes:
            out["avg_effective_tax_rate"] = np.mean(self.periodic_bracket_tax.all_effective_tax_rates)
            out["total_collected_taxes"] = float(self.periodic_bracket_tax.total_collected_taxes)

            # Indices of richest and poorest agents.
            agent_coin_endows = np.array(
                [agent.total_endowment("Coin") for agent in self.world.agents]
            )
            idx_poor = np.argmin(agent_coin_endows)
            idx_rich = np.argmax(agent_coin_endows)

            tax_days = self.periodic_bracket_tax.taxes[(self.period - 1) :: self.period]
            for i, tag in zip([idx_poor, idx_rich], ["poorest", "richest"]):
                total_income = np.maximum(
                    0, [tax_day[str(i)]["income"] for tax_day in tax_days]
                ).sum()
                total_tax_paid = np.sum(
                    [tax_day[str(i)]["tax_paid"] for tax_day in tax_days]
                )
                # Report the overall tax rate over the episode for the richest and poorest agents.
                out["avg_tax_rate/{}".format(tag)] = total_tax_paid / np.maximum(
                    0.001, total_income
                )

            if self.tax_model == "saez":
                # Include the running estimate of elasticity.
                out["saez/estimated_elasticity"] = self.periodic_bracket_tax.elas_tm1

        return out
    
    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Reset tax scheduling trackers and re-sample agents' vote and invest choices: ["Wood", "Stone", "Iron"].
        """
        
        self.periodic_bracket_tax.curr_rate_indices = [np.random.permutation(21)[0] for _ in range(self.n_brackets)]

        self.periodic_bracket_tax.tax_cycle_pos = 1
        self.periodic_bracket_tax.last_coin = [
            float(agent.total_endowment("Coin")) for agent in self.world.agents
        ]
        self.periodic_bracket_tax.last_income = [0 for _ in range(self.n_agents)]
        self.periodic_bracket_tax.last_marginal_rate = [0 for _ in range(self.n_agents)]
        self.periodic_bracket_tax.last_effective_tax_rate = [0 for _ in range(self.n_agents)]

        self.periodic_bracket_tax._curr_rates_obs = np.array(self.periodic_bracket_tax.curr_marginal_rates)
        self.periodic_bracket_tax._last_income_obs = np.array(self.periodic_bracket_tax.last_income) / self.period
        self.periodic_bracket_tax._last_income_obs_sorted = self.periodic_bracket_tax._last_income_obs[
            np.argsort(self.periodic_bracket_tax._last_income_obs)
        ]

        self.periodic_bracket_tax.taxes = []
        self.periodic_bracket_tax.total_collected_taxes = 0
        self.periodic_bracket_tax.all_effective_tax_rates = []
        self.periodic_bracket_tax._schedules = {"{:03d}".format(int(r)): [] for r in self.periodic_bracket_tax.bracket_cutoffs}
        self.periodic_bracket_tax._occupancy = {"{:03d}".format(int(r)): 0 for r in self.periodic_bracket_tax.bracket_cutoffs}
        self.periodic_bracket_tax._planner_masks = None

        if self.tax_model == "saez":
            self.periodic_bracket_tax.curr_bracket_tax_rates = np.array(self.periodic_bracket_tax.running_avg_tax_rates)
        
        self.votesinvests = []

    def get_dense_log(self):
        """
        Log votes and invests and taxes.

        Returns:
            taxes (list): A list of tax collections. Each entry corresponds to a single timestep. Entries are empty 
            except for timesteps where a tax period ended and taxes were collected. For those timesteps, each entry
            contains the tax schedule, each agent's reported income, tax paid, and redistribution received.
                
            votesinvests (list): A list of vote and invest events. Each entry corresponds to a single timestep and 
            contains a description of any vote and invest that occurred on that timestep.
        """
        
        if self.disable_taxes:
            return self.votesinvests
        else:
            return self.periodic_bracket_tax.taxes, self.votesinvests
   
        
@component_registry.add
class AgentVotesPlannerInvestsResources(BaseComponent):
    """
    Allows mobile agents to vote for and the planner to invest on three resources (wood, stone, and iron).
    
    Args:
        vote_count_method: The method which the votes of the agents are counted 
            by the central social planner.
            Plurality voting: Each voter casts a single vote. The candidate with 
                the most votes is selected.
            Majority voting: Each voter ranks all items. The overall rank of each 
                item is determined based on all pairwise comparisons.
            Borda voting (default): Each voter submits a full ordering on the 
                candidates. This ordering contributes points to each candidate; 
                if there are n candidates, it contributes n-1 points to the 
                highest ranked candidate, n-2  points to the second highest, 
                and so on; it contributes no points to the lowest ranked candidate. 
                The winners are those whose total sum of points from all the 
                voters is maximal.
        institution_type: Institution type which could be one of the followings:
            Inclusive (default): The central social planner collects the due taxes of all 
                agents and counts the votes of all agents equally.
            Extractive: The central social planner collects the due taxes of all 
                agents while only counts the votes of the top 60% richest of agents.
            Arbitrary: The central social planner collects the due taxes of all agents 
                while only counts the votes of 60% of the agents selected randomly.
        disable_taxes (bool): Whether to disable any tax collection, effectively
            enforcing that tax rates are always 0. Useful for removing taxes without
            changing the observation space. Default is False (taxes enabled).
        tax_model (str): Which tax model to use for setting taxes.
            "model_wrapper" (default) uses the actions of the planner agent;
            "saez" uses an adaptation of the theoretical optimal taxation formula
            derived in https://www.nber.org/papers/w7628.
            "us-federal-single-filer-2018-scaled" uses US federal tax rates from 2018;
            "fixed-bracket-rates" uses the rates supplied in fixed_bracket_rates.
        period (int): Length of a tax period in environment timesteps. Taxes are
            updated at the start of each period and collected/redistributed at the
            end of each period. Must be > 0. Default is 100 timesteps.
        rate_min (float): Minimum tax rate within a bracket. Must be >= 0 (default).
        rate_max (float): Maximum tax rate within a bracket. Must be <= 1 (default).
        rate_disc (float): (Only applies for "model_wrapper") the interval separating
            discrete tax rates that the planner can select. Default of 0.05 means,
            for example, the planner can select among [0.0, 0.05, 0.10, ... 1.0].
            Must be > 0 and < 1.
        n_brackets (int): How many tax brackets to use. Must be >=2. Default is 5.
        top_bracket_cutoff (float): The income at the left end of the last tax
            bracket. Must be >= 10. Default is 100 coin.
        usd_scaling (float): Scale by which to divide the US Federal bracket cutoffs
            when using bracket_spacing = "us-federal". Must be > 0. Default is 1000.
        bracket_spacing (str): How bracket cutoffs should be spaced.
            "us-federal" (default) uses scaled cutoffs from the 2018 US Federal
                taxes, with scaling set by usd_scaling (ignores n_brackets and
                top_bracket_cutoff);
            "linear" linearly spaces the n_bracket cutoffs between 0 and
                top_bracket_cutoff;
            "log" is similar to "linear" but with logarithmic spacing.
        fixed_bracket_rates (list): Required if tax_model=="fixed-bracket-rates". A
            list of fixed marginal rates to use for each bracket. Length must be
            equal to the number of brackets (7 for "us-federal" spacing, n_brackets
            otherwise).
        pareto_weight_type (str): Type of Pareto weights to use when computing tax
            rates using the Saez formula. "inverse_income" (default) uses 1/z;
            "uniform" uses 1.
        saez_fixed_elas (float, optional): If supplied, this value will be used as
            the elasticity estimate when computing tax rates using the Saez formula.
            If not given (default), elasticity will be estimated empirically.
        tax_annealing_schedule (list, optional): A length-2 list of
            [tax_annealing_warmup, tax_annealing_slope] describing the tax annealing
            schedule. See annealed_tax_mask function for details. Default behavior is
            no tax annealing.     
    """
    
    name = "AgentVotesPlannerInvestsResources"
    component_type = "VoteInvest"
    required_entities = ["Wood", "Stone", "Iron", "Coin", "VoteInvest"]
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]
    
    def __init__(
        self,
        *base_component_args,
        
        vote_count_method='Borda',
        institution_type='Inclusive',
        
        disable_taxes=False,
        tax_model='model_wrapper',
        period=100,
        rate_min=0.0,
        rate_max=1.0,
        rate_disc=0.05,
        n_brackets=7,
        top_bracket_cutoff=100,
        usd_scaling=1000.0,
        bracket_spacing='us-federal',
        fixed_bracket_rates=None,
        pareto_weight_type='inverse_income',
        saez_fixed_elas=None,
        tax_annealing_schedule=None,
        
        **base_component_kwargs
    ):
        
        super().__init__(*base_component_args, **base_component_kwargs)
        
        # How to set vote count methods.
        self.vote_count_method = vote_count_method
        assert self.vote_count_method in [
            "Plurality",
            "Majority",
            "Borda",
        ]
        
        # How to set institution types.
        self.institution_type = institution_type
        assert self.institution_type in [
            "Inclusive",
            "Extractive",
            "Arbitrary",
        ]
        
        ### From redistribution module

        self.disable_taxes=disable_taxes
        self.tax_model=tax_model
        self.period=period
        self.rate_min=rate_min
        self.rate_max=rate_max
        self.rate_disc=rate_disc
        self.n_brackets=n_brackets
        self.top_bracket_cutoff=top_bracket_cutoff
        self.usd_scaling=usd_scaling
        self.bracket_spacing=bracket_spacing
        self.fixed_bracket_rates=fixed_bracket_rates
        self.pareto_weight_type=pareto_weight_type
        self.saez_fixed_elas=saez_fixed_elas
        self.tax_annealing_schedule=tax_annealing_schedule
        
        self.periodic_bracket_tax = PeriodicBracketTax(
            world=self.world,
            episode_length=self.episode_length,
            disable_taxes=self.disable_taxes,
            tax_model=self.tax_model,
            period=self.period,
            rate_min=self.rate_min,
            rate_max=self.rate_max,
            rate_disc=self.rate_disc,
            n_brackets=self.n_brackets,
            top_bracket_cutoff=self.top_bracket_cutoff,
            usd_scaling=self.usd_scaling,
            bracket_spacing=self.bracket_spacing,
            fixed_bracket_rates=self.fixed_bracket_rates,
            pareto_weight_type=self.pareto_weight_type,
            saez_fixed_elas=self.saez_fixed_elas,
            tax_annealing_schedule=self.tax_annealing_schedule,
        )
        
        ###
        
        self.votesinvests = []

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.
        
        Add six actions (voting for and investing on three resources using one of six possibilities) for mobile agents.
        
        Also, if using the "model_wrapper" tax model and taxes are enabled, the planner's action space includes an action 
        subspace for each of the tax brackets. Each such action space has as many actions as there are discretized tax rates.
        """
        
        # This component adds six actions that mobile agents can take when vote for and invest on three different resources.
        if agent_cls_name == "BasicMobileAgent":
            return 6
        
        # Also, the planner takes actions through tax scheduling.
        if agent_cls_name == "BasicPlanner":
            if self.tax_model == "model_wrapper" and not self.disable_taxes:
                # For every bracket, the planner can select one of the discretized tax rates.
                return [
                    ("TaxIndexBracket_{:03d}".format(int(r)), self.periodic_bracket_tax.n_disc_rates)
                    for r in self.periodic_bracket_tax.bracket_cutoffs
                ]
    
    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        For mobile agents and the planner, add a state field indicating the vote counting method.
        """
        
        if agent_cls_name not in self.agent_subclasses:
            return {}
        else:
            return {"vote_count_method": self.vote_count_method}
        
        raise NotImplementedError

    def component_step(self): 
        """
        See base_component.py for detailed description.
        
        On the first day of each tax period, update taxes. On the last day, enact them.
        
        Also, rank three resources (wood, stone, and iron) and using the collected taxes to invest on them.
        """
        
        # 1. On the first day of a new tax period: set up the taxes for this period.
        if self.periodic_bracket_tax.tax_cycle_pos == 1:
            if self.tax_model == "model_wrapper":
                self.periodic_bracket_tax.set_new_period_rates_model()
    
            if self.tax_model == "saez":
                self.periodic_bracket_tax.compute_and_set_new_period_rates_from_saez_formula()
    
            self.periodic_bracket_tax._curr_rates_obs = np.array(self.periodic_bracket_tax.curr_marginal_rates)
    
        # 2. On the last day of the tax period: get $-taxes and update agent endowments.
        if self.periodic_bracket_tax.tax_cycle_pos >= self.period:
            self.periodic_bracket_tax.enact_taxes()
            self.periodic_bracket_tax.tax_cycle_pos = 0
    
        else:
            self.periodic_bracket_tax.taxes.append([])
    
        # Increment timestep.
        self.periodic_bracket_tax.tax_cycle_pos += 1
        
        if self.institution_type == "Extractive":
            number_top_50percent_rich_agents = int(0.5 * self.n_agents)
            
            agent_coin_endows = np.array([agent.total_endowment("Coin") for agent in self.world.agents])
            
            top_50percent_rich_agents = np.argsort(agent_coin_endows)[self.n_agents - number_top_50percent_rich_agents:]
            
        elif self.institution_type == "Arbitrary":
            number_random_50percent_agents = int(0.5 * self.n_agents)
            
            random_50percent_agents = np.random.permutation(self.n_agents)[:number_random_50percent_agents]
            
        else:
            pass
        
        voteinvest = []
        investments = np.zeros(3, float) 

        matrix_order = np.zeros((6, 3), int)
        resource_array = ["Wood", "Stone", "Iron"]

        matrix_order[0, :] = [2, 1, 0]
        matrix_order[1, :] = [2, 0, 1]
        matrix_order[2, :] = [1, 2, 0]
        matrix_order[3, :] = [1, 0, 2]
        matrix_order[4, :] = [0, 2, 1]
        matrix_order[5, :] = [0, 1, 2]
 
        for agent in self.world.get_random_order_agents():
            action = agent.get_component_action(self.name)
            
            counter_vote_actions_plurality = np.zeros(3, int)
            counter_vote_actions_majority_borda = np.zeros(3, int)
            
            if self.institution_type == "Extractive":
                if int(agent.idx) in top_50percent_rich_agents:
                    agent_flag = True
                else:
                    agent_flag = False
                    
            elif self.institution_type == "Arbitrary":
                if int(agent.idx) in random_50percent_agents:
                    agent_flag = True
                else:
                    agent_flag = False
            
            else:
                agent_flag = True

            action_flag = False

            # NO-OP!
            if action == 0 or action is None:
                action_flag = True

                pass
            
            # Vote for and invest on wood, stone, and iron!
            for i in range(6):
                if action_flag == True:
                    break
                
                if action == i + 1:
                    if agent_flag == True:
                        counter_vote_actions_plurality[int(i / 2)] = counter_vote_actions_plurality[int(i / 2)] + 1
                    
                        counter_vote_actions_majority_borda[0] = counter_vote_actions_majority_borda[0] + matrix_order[i, 0]
                        counter_vote_actions_majority_borda[1] = counter_vote_actions_majority_borda[1] + matrix_order[i, 1]
                        counter_vote_actions_majority_borda[2] = counter_vote_actions_majority_borda[2] + matrix_order[i, 2]

                    action_flag = True
            
            if action_flag == False:
                # raise ValueError
                pass
        
        investment = self.periodic_bracket_tax.get_metrics()["total_collected_taxes"]
        
        if self.vote_count_method == "Plurality":
            if counter_vote_actions_plurality[0] > counter_vote_actions_plurality[1] and \
               counter_vote_actions_plurality[0] > counter_vote_actions_plurality[2]: 
                
                investments = investments + [investment, 0, 0]
                
                for agent in self.world.get_random_order_agents():
                    agent.state["endogenous"]["VoteInvest"] = investments
            
            elif counter_vote_actions_plurality[1] > counter_vote_actions_plurality[0] and \
                 counter_vote_actions_plurality[1] > counter_vote_actions_plurality[2]:

                investments = investments + [0, investment, 0]
                
                for agent in self.world.get_random_order_agents():
                    agent.state["endogenous"]["VoteInvest"] = investments
            
            elif counter_vote_actions_plurality[2] > counter_vote_actions_plurality[0] and \
                 counter_vote_actions_plurality[2] > counter_vote_actions_plurality[1]:

                investments = investments + [0, 0, investment]
                
                for agent in self.world.get_random_order_agents():  
                    agent.state["endogenous"]["VoteInvest"] = investments

        elif self.vote_count_method == "Majority":
            for i in range(6):
                if (np.argsort(counter_vote_actions_majority_borda) == matrix_order[i, :]).all():
                    investments = investments + np.array([(matrix_order[i, 0] + 1) * investment / 6, (matrix_order[i, 1] + 1) * investment / 6, (matrix_order[i, 2] + 1) * investment / 6])
                
                    for agent in self.world.get_random_order_agents():
                        agent.state["endogenous"]["VoteInvest"] = investments
             
        elif self.vote_count_method == "Borda":
            investments = investments + np.array([counter_vote_actions_majority_borda[0] / (counter_vote_actions_majority_borda[0] + counter_vote_actions_majority_borda[1] + counter_vote_actions_majority_borda[2] + epsilon) * investment,
                                                  counter_vote_actions_majority_borda[1] / (counter_vote_actions_majority_borda[0] + counter_vote_actions_majority_borda[1] + counter_vote_actions_majority_borda[2] + epsilon) * investment,
                                                  counter_vote_actions_majority_borda[2] / (counter_vote_actions_majority_borda[0] + counter_vote_actions_majority_borda[1] + counter_vote_actions_majority_borda[2] + epsilon) * investment])
                
            for agent in self.world.get_random_order_agents():
                agent.state["endogenous"]["VoteInvest"] = investments

        for i in range(6):
            if (np.argsort(counter_vote_actions_majority_borda) == matrix_order[i, :]).all():
                voteinvest.append(
                    {
                       "vote_count_method": self.vote_count_method,
                       "institution_type": self.institution_type,
                       "agents": ["Agent", "Planner"],
                       "vote": [resource_array[matrix_order[i, 2]], resource_array[matrix_order[i, 1]], resource_array[matrix_order[i, 0]]],
                       "invest": investments,
                    }
                )
               
        self.votesinvests.append(voteinvest)
        
        return investments

    def generate_observations(self):
        """
        See base_component.py for detailed description.
        
        Agents observe where in the tax period cycle they are, information about the last period's incomes, and the current 
        marginal tax rates, including the marginal rate that will apply to their next unit of income.
        
        Also, the planner observes the same type of information, but for all the agents. It also sees, for each agent, their 
        marginal tax rate and reported income from the previous tax period.
        
        Moreover, agents observe their choice of vote and investment. The planner does not observe any vote and investment actions.
        """
        
        is_tax_day = float(self.periodic_bracket_tax.tax_cycle_pos >= self.period)
        is_first_day = float(self.periodic_bracket_tax.tax_cycle_pos == 1)
        tax_phase = self.periodic_bracket_tax.tax_cycle_pos / self.period
        
        obs = dict()
        
        obs[self.world.planner.idx] = dict(
            is_tax_day=is_tax_day,
            is_first_day=is_first_day,
            tax_phase=tax_phase,
            last_incomes=self.periodic_bracket_tax._last_income_obs_sorted,
            curr_rates=self.periodic_bracket_tax._curr_rates_obs,
        )
        
        for agent in self.world.agents:           
            i = agent.idx
            k = str(i)
        
            curr_marginal_rate = self.periodic_bracket_tax.marginal_rate(
                agent.total_endowment("Coin") - self.periodic_bracket_tax.last_coin[i]
            )
        
            obs[k] = dict(
                is_tax_day=is_tax_day,
                is_first_day=is_first_day,
                tax_phase=tax_phase,
                last_incomes=self.periodic_bracket_tax._last_income_obs_sorted,
                curr_rates=self.periodic_bracket_tax._curr_rates_obs,
                marginal_rate=curr_marginal_rate,
                vote_invest_action=agent.state["endogenous"]["VoteInvest"],
            )
        
            obs["p" + k] = dict(
                last_income=self.periodic_bracket_tax._last_income_obs[i],
                last_marginal_rate=self.periodic_bracket_tax.last_marginal_rate[i],
                curr_marginal_rate=curr_marginal_rate,
            )
        
        return obs

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.
        
        Tax scheduling masks only apply to the planner if tax_model == "model_wrapper" and taxes are enabled. 
        All tax actions are masked (so, only NO-OPs can be sampled) on all timesteps except when self.tax_cycle_pos==1 
        (meaning a new tax period is starting). When self.tax_cycle_pos==1, tax actions are masked in order to enforce 
        any tax annealing.
        
        Moreover, agents can always vote.
        """
        
        if (
            completions != self.periodic_bracket_tax._last_completions
            and self.tax_annealing_schedule is not None
        ):
            self.periodic_bracket_tax._last_completions = int(completions)
            self.periodic_bracket_tax._annealed_rate_max = self.periodic_bracket_tax.annealed_tax_limit(
                completions,
                self.periodic_bracket_tax._annealing_warmup,
                self.periodic_bracket_tax._annealing_slope,
                self.rate_max,
            )

        if self.disable_taxes:
            return {}

        if self.tax_model == "model_wrapper":
            # No annealing. Generate masks using default method.
            if self.tax_annealing_schedule is None:
                if self.periodic_bracket_tax._planner_masks is None:
                    masks = super().generate_masks(completions=completions)
                    self.periodic_bracket_tax._planner_masks = dict(
                        new_taxes=deepcopy(masks[self.world.planner.idx]),
                        zeros={
                            k: np.zeros_like(v)
                            for k, v in masks[self.world.planner.idx].items()
                        },
                    )

                # No need to recompute. Use the cached masks.
                masks = dict()
                if self.periodic_bracket_tax.tax_cycle_pos != 1 or self.disable_taxes:
                    # Apply zero masks for any timestep where taxes are not going to be updated.
                    masks[self.world.planner.idx] = self.periodic_bracket_tax._planner_masks["zeros"]
                else:
                    masks[self.world.planner.idx] = self.periodic_bracket_tax._planner_masks["new_taxes"]

            # Doing annealing.
            else:
                # Figure out what the masks should be this episode.
                if self.periodic_bracket_tax._planner_masks is None:
                    planner_masks = {
                        k: self.periodic_bracket_tax.annealed_tax_mask(
                            completions,
                            self.periodic_bracket_tax._annealing_warmup,
                            self.periodic_bracket_tax._annealing_slope,
                            tax_values,
                        )
                        for k, tax_values in self.periodic_bracket_tax._planner_tax_val_dict.items()
                    }
                    self.periodic_bracket_tax._planner_masks = dict(
                        new_taxes=deepcopy(planner_masks),
                        zeros={k: np.zeros_like(v) for k, v in planner_masks.items()},
                    )

                # No need to recompute. Use the cached masks.
                masks = dict()
                if self.periodic_bracket_tax.tax_cycle_pos != 1 or self.disable_taxes:
                    # Apply zero masks for any timestep where taxes are not going to be updated.
                    masks[self.world.planner.idx] = self.periodic_bracket_tax._planner_masks["zeros"]
                else:
                    masks[self.world.planner.idx] = self.periodic_bracket_tax._planner_masks["new_taxes"]

        # We are not using a learned planner. Generate masks by the default method.
        else:
            masks = super().generate_masks(completions=completions)
            
        for agent in self.world.agents:
            masks[agent.idx] = np.array([1, 1, 1, 1, 1, 1])

        return masks

    def get_metrics(self):
        """
        Metrics that capture what happened through this component.

        Return metrics related to bracket rates, bracket occupancy, and tax collection.
        """
        
        out = dict()

        n_observed_incomes = np.maximum(1, np.sum(list(self.periodic_bracket_tax._occupancy.values())))
        for c in self.periodic_bracket_tax.bracket_cutoffs:
            k = "{:03d}".format(int(c))
            out["avg_bracket_rate/{}".format(k)] = np.mean(self.periodic_bracket_tax._schedules[k])
            out["bracket_occupancy/{}".format(k)] = (
                self.periodic_bracket_tax._occupancy[k] / n_observed_incomes
            )

        if not self.disable_taxes:
            out["avg_effective_tax_rate"] = np.mean(self.periodic_bracket_tax.all_effective_tax_rates)
            out["total_collected_taxes"] = float(self.periodic_bracket_tax.total_collected_taxes)

            # Indices of richest and poorest agents.
            agent_coin_endows = np.array(
                [agent.total_endowment("Coin") for agent in self.world.agents]
            )
            idx_poor = np.argmin(agent_coin_endows)
            idx_rich = np.argmax(agent_coin_endows)

            tax_days = self.periodic_bracket_tax.taxes[(self.period - 1) :: self.period]
            for i, tag in zip([idx_poor, idx_rich], ["poorest", "richest"]):
                total_income = np.maximum(
                    0, [tax_day[str(i)]["income"] for tax_day in tax_days]
                ).sum()
                total_tax_paid = np.sum(
                    [tax_day[str(i)]["tax_paid"] for tax_day in tax_days]
                )
                # Report the overall tax rate over the episode for the richest and poorest agents.
                out["avg_tax_rate/{}".format(tag)] = total_tax_paid / np.maximum(
                    0.001, total_income
                )

            if self.tax_model == "saez":
                # Include the running estimate of elasticity.
                out["saez/estimated_elasticity"] = self.periodic_bracket_tax.elas_tm1

        return out
    
    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Reset tax scheduling trackers and re-sample agents' vote and invest choices: ["Wood", "Stone", "Iron"].
        """
        
        self.periodic_bracket_tax.curr_rate_indices = [np.random.permutation(21)[0] for _ in range(self.n_brackets)]

        self.periodic_bracket_tax.tax_cycle_pos = 1
        self.periodic_bracket_tax.last_coin = [
            float(agent.total_endowment("Coin")) for agent in self.world.agents
        ]
        self.periodic_bracket_tax.last_income = [0 for _ in range(self.n_agents)]
        self.periodic_bracket_tax.last_marginal_rate = [0 for _ in range(self.n_agents)]
        self.periodic_bracket_tax.last_effective_tax_rate = [0 for _ in range(self.n_agents)]

        self.periodic_bracket_tax._curr_rates_obs = np.array(self.periodic_bracket_tax.curr_marginal_rates)
        self.periodic_bracket_tax._last_income_obs = np.array(self.periodic_bracket_tax.last_income) / self.period
        self.periodic_bracket_tax._last_income_obs_sorted = self.periodic_bracket_tax._last_income_obs[
            np.argsort(self.periodic_bracket_tax._last_income_obs)
        ]

        self.periodic_bracket_tax.taxes = []
        self.periodic_bracket_tax.total_collected_taxes = 0
        self.periodic_bracket_tax.all_effective_tax_rates = []
        self.periodic_bracket_tax._schedules = {"{:03d}".format(int(r)): [] for r in self.periodic_bracket_tax.bracket_cutoffs}
        self.periodic_bracket_tax._occupancy = {"{:03d}".format(int(r)): 0 for r in self.periodic_bracket_tax.bracket_cutoffs}
        self.periodic_bracket_tax._planner_masks = None

        if self.tax_model == "saez":
            self.periodic_bracket_tax.curr_bracket_tax_rates = np.array(self.periodic_bracket_tax.running_avg_tax_rates)
             
        self.votesinvests = []
        
    def get_dense_log(self):
        """
        Log votes and invests and taxes.

        Returns:
            taxes (list): A list of tax collections. Each entry corresponds to a single timestep. Entries are empty 
            except for timesteps where a tax period ended and taxes were collected. For those timesteps, each entry
            contains the tax schedule, each agent's reported income, tax paid, and redistribution received.
                
            votesinvests (list): A list of vote and invest events. Each entry corresponds to a single timestep and 
            contains a description of any vote and invest that occurred on that timestep.
        """
        
        if self.disable_taxes:
            return self.votesinvests
        else:
            return self.periodic_bracket_tax.taxes, self.votesinvests
    
    
@component_registry.add
class PlannerVotesPlannerInvestsResources(BaseComponent):
    """
    Allows the planner to vote for and to invest on three resources (wood, stone, and iron).
    
    Args:
        disable_taxes (bool): Whether to disable any tax collection, effectively
            enforcing that tax rates are always 0. Useful for removing taxes without
            changing the observation space. Default is False (taxes enabled).
        tax_model (str): Which tax model to use for setting taxes.
            "model_wrapper" (default) uses the actions of the planner agent;
            "saez" uses an adaptation of the theoretical optimal taxation formula
            derived in https://www.nber.org/papers/w7628.
            "us-federal-single-filer-2018-scaled" uses US federal tax rates from 2018;
            "fixed-bracket-rates" uses the rates supplied in fixed_bracket_rates.
        period (int): Length of a tax period in environment timesteps. Taxes are
            updated at the start of each period and collected/redistributed at the
            end of each period. Must be > 0. Default is 100 timesteps.
        rate_min (float): Minimum tax rate within a bracket. Must be >= 0 (default).
        rate_max (float): Maximum tax rate within a bracket. Must be <= 1 (default).
        rate_disc (float): (Only applies for "model_wrapper") the interval separating
            discrete tax rates that the planner can select. Default of 0.05 means,
            for example, the planner can select among [0.0, 0.05, 0.10, ... 1.0].
            Must be > 0 and < 1.
        n_brackets (int): How many tax brackets to use. Must be >=2. Default is 5.
        top_bracket_cutoff (float): The income at the left end of the last tax
            bracket. Must be >= 10. Default is 100 coin.
        usd_scaling (float): Scale by which to divide the US Federal bracket cutoffs
            when using bracket_spacing = "us-federal". Must be > 0. Default is 1000.
        bracket_spacing (str): How bracket cutoffs should be spaced.
            "us-federal" (default) uses scaled cutoffs from the 2018 US Federal
                taxes, with scaling set by usd_scaling (ignores n_brackets and
                top_bracket_cutoff);
            "linear" linearly spaces the n_bracket cutoffs between 0 and
                top_bracket_cutoff;
            "log" is similar to "linear" but with logarithmic spacing.
        fixed_bracket_rates (list): Required if tax_model=="fixed-bracket-rates". A
            list of fixed marginal rates to use for each bracket. Length must be
            equal to the number of brackets (7 for "us-federal" spacing, n_brackets
            otherwise).
        pareto_weight_type (str): Type of Pareto weights to use when computing tax
            rates using the Saez formula. "inverse_income" (default) uses 1/z;
            "uniform" uses 1.
        saez_fixed_elas (float, optional): If supplied, this value will be used as
            the elasticity estimate when computing tax rates using the Saez formula.
            If not given (default), elasticity will be estimated empirically.
        tax_annealing_schedule (list, optional): A length-2 list of
            [tax_annealing_warmup, tax_annealing_slope] describing the tax annealing
            schedule. See annealed_tax_mask function for details. Default behavior is
            no tax annealing.     
    """
    
    name = "PlannerVotesPlannerInvestsResources"
    component_type = "VoteInvest"
    required_entities = ["Wood", "Stone", "Iron", "Coin", "VoteInvest"]
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]
    
    def __init__(
        self,
        *base_component_args,
        
        disable_taxes=False,
        tax_model='model_wrapper',
        period=100,
        rate_min=0.0,
        rate_max=1.0,
        rate_disc=0.05,
        n_brackets=7,
        top_bracket_cutoff=100,
        usd_scaling=1000.0,
        bracket_spacing='us-federal',
        fixed_bracket_rates=None,
        pareto_weight_type='inverse_income',
        saez_fixed_elas=None,
        tax_annealing_schedule=None,
        
        **base_component_kwargs
    ):
        
        super().__init__(*base_component_args, **base_component_kwargs)
        
        ### From redistribution module

        self.disable_taxes=disable_taxes
        self.tax_model=tax_model
        self.period=period
        self.rate_min=rate_min
        self.rate_max=rate_max
        self.rate_disc=rate_disc
        self.n_brackets=n_brackets
        self.top_bracket_cutoff=top_bracket_cutoff
        self.usd_scaling=usd_scaling
        self.bracket_spacing=bracket_spacing
        self.fixed_bracket_rates=fixed_bracket_rates
        self.pareto_weight_type=pareto_weight_type
        self.saez_fixed_elas=saez_fixed_elas
        self.tax_annealing_schedule=tax_annealing_schedule
        
        self.periodic_bracket_tax = PeriodicBracketTax(
            world=self.world,
            episode_length=self.episode_length,
            disable_taxes=self.disable_taxes,
            tax_model=self.tax_model,
            period=self.period,
            rate_min=self.rate_min,
            rate_max=self.rate_max,
            rate_disc=self.rate_disc,
            n_brackets=self.n_brackets,
            top_bracket_cutoff=self.top_bracket_cutoff,
            usd_scaling=self.usd_scaling,
            bracket_spacing=self.bracket_spacing,
            fixed_bracket_rates=self.fixed_bracket_rates,
            pareto_weight_type=self.pareto_weight_type,
            saez_fixed_elas=self.saez_fixed_elas,
            tax_annealing_schedule=self.tax_annealing_schedule,
        )
        
        ###
        
        self.votesinvests = []
        
    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.
        
        Add six actions (voting for and investing on three resources using one of six possibilities) for the planner.
        
        Also, if using the "model_wrapper" tax model and taxes are enabled, the planner's action space includes an action 
        subspace for each of the tax brackets. Each such action space has as many actions as there are discretized tax rates.
        """
        
        # The planner takes actions through tax scheduling and through vote and investment
        if agent_cls_name == "BasicPlanner":
            if self.tax_model == "model_wrapper" and not self.disable_taxes:
                # For every bracket, the planner can select one of the discretized tax rates.
                return [
                    ("TaxIndexBracket_{:03d}".format(int(r)), self.periodic_bracket_tax.n_disc_rates)
                    for r in self.periodic_bracket_tax.bracket_cutoffs
                ] + [("VoteInvest", 6)]
     
    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        For mobile agents and the planner, add a state field indicating the vote counting method.
        """
        
        if agent_cls_name not in self.agent_subclasses:
            return {}
        else:
            return {"vote_count_method": "Not required!"}
        
        raise NotImplementedError

    def component_step(self):
        """
        See base_component.py for detailed description.
        
        On the first day of each tax period, update taxes. On the last day, enact them.
        
        Also, rank three resources (wood, stone, and iron) and using the collected taxes to invest on them.
        """
        
        # 1. On the first day of a new tax period: set up the taxes for this period.
        if self.periodic_bracket_tax.tax_cycle_pos == 1:
            if self.tax_model == "model_wrapper":
                self.periodic_bracket_tax.set_new_period_rates_model()
    
            if self.tax_model == "saez":
                self.periodic_bracket_tax.compute_and_set_new_period_rates_from_saez_formula()
    
            self.periodic_bracket_tax._curr_rates_obs = np.array(self.periodic_bracket_tax.curr_marginal_rates)
    
        # 2. On the last day of the tax period: get $-taxes and update agent endowments.
        if self.periodic_bracket_tax.tax_cycle_pos >= self.period:
            self.periodic_bracket_tax.enact_taxes()
            self.periodic_bracket_tax.tax_cycle_pos = 0
    
        else:
            self.periodic_bracket_tax.taxes.append([])
    
        # Increment timestep.
        self.periodic_bracket_tax.tax_cycle_pos += 1
        
        action = self.world.planner.get_component_action(self.name)[self.n_brackets]
        
        investment = self.periodic_bracket_tax.get_metrics()["total_collected_taxes"]

        voteinvest = []
        investments = np.zeros(3, float)

        matrix_order = np.zeros((6, 3), int)
        resource_array = ["Wood", "Stone", "Iron"]

        matrix_order[0, :] = [3, 2, 1]
        matrix_order[1, :] = [3, 1, 2]
        matrix_order[2, :] = [2, 3, 1]
        matrix_order[3, :] = [2, 1, 3]
        matrix_order[4, :] = [1, 3, 2]
        matrix_order[5, :] = [1, 2, 3]

        action_flag = False

        # NO-OP!
        if action == 0 or action is None:
            action_flag = True

            pass
        
        # Vote for and invest on wood, stone, iron, and soil!
        for i in range(6):
            if action_flag == True:
                break
            
            if action == i + 1:
                investments = investments + np.array([matrix_order[i, 0] * investment / 6, matrix_order[i, 1] * investment / 6, matrix_order[i, 2] * investment / 6])
            
                self.world.planner.state["endogenous"]["VoteInvest"] = investments
            
                voteinvest.append(
                    {
                        "agents": ["Planner", "Planner"],
                        "vote": [resource_array[matrix_order[i, 2] - 1], resource_array[matrix_order[i, 1] - 1], resource_array[matrix_order[i, 0] - 1]],
                        "invest": investments,
                    }
                )

                action_flag = True
            
        if action_flag == False:
            # raise ValueError
            pass
            
        self.votesinvests.append(voteinvest)
        
        return investments

    def generate_observations(self):
        """
        See base_component.py for detailed description.
        
        Agents observe where in the tax period cycle they are, information about the last period's incomes, and the current 
        marginal tax rates, including the marginal rate that will apply to their next unit of income.
        
        Also, the planner observes the same type of information, but for all the agents. It also sees, for each agent, their 
        marginal tax rate and reported income from the previous tax period.
        
        Moreover, the planner observes its choice of vote and investment. The agents do not observe any vote and investment actions.
        """
        
        is_tax_day = float(self.periodic_bracket_tax.tax_cycle_pos >= self.period)
        is_first_day = float(self.periodic_bracket_tax.tax_cycle_pos == 1)
        tax_phase = self.periodic_bracket_tax.tax_cycle_pos / self.period
        
        obs = dict()
        
        obs[self.world.planner.idx] = dict(
            is_tax_day=is_tax_day,
            is_first_day=is_first_day,
            tax_phase=tax_phase,
            last_incomes=self.periodic_bracket_tax._last_income_obs_sorted,
            curr_rates=self.periodic_bracket_tax._curr_rates_obs,
            vote_invest_action=self.world.planner.state["endogenous"]["VoteInvest"],
        )
        
        for agent in self.world.agents:           
            i = agent.idx
            k = str(i)
        
            curr_marginal_rate = self.periodic_bracket_tax.marginal_rate(
                agent.total_endowment("Coin") - self.periodic_bracket_tax.last_coin[i]
            )
        
            obs[k] = dict(
                is_tax_day=is_tax_day,
                is_first_day=is_first_day,
                tax_phase=tax_phase,
                last_incomes=self.periodic_bracket_tax._last_income_obs_sorted,
                curr_rates=self.periodic_bracket_tax._curr_rates_obs,
                marginal_rate=curr_marginal_rate,
            )
        
            obs["p" + k] = dict(
                last_income=self.periodic_bracket_tax._last_income_obs[i],
                last_marginal_rate=self.periodic_bracket_tax.last_marginal_rate[i],
                curr_marginal_rate=curr_marginal_rate,
            )
        
        return obs

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.
        
        Tax scheduling masks only apply to the planner if tax_model == "model_wrapper" and taxes are enabled. 
        All tax actions are masked (so, only NO-OPs can be sampled) on all timesteps except when self.tax_cycle_pos==1 
        (meaning a new tax period is starting). When self.tax_cycle_pos==1, tax actions are masked in order to enforce 
        any tax annealing.
        
        Moreover, the planner can always vote.
        """
        
        if (
            completions != self.periodic_bracket_tax._last_completions
            and self.tax_annealing_schedule is not None
        ):
            self.periodic_bracket_tax._last_completions = int(completions)
            self.periodic_bracket_tax._annealed_rate_max = self.periodic_bracket_tax.annealed_tax_limit(
                completions,
                self.periodic_bracket_tax._annealing_warmup,
                self.periodic_bracket_tax._annealing_slope,
                self.rate_max,
            )

        if self.disable_taxes:
            return {}

        if self.tax_model == "model_wrapper":
            # No annealing. Generate masks using default method.
            if self.tax_annealing_schedule is None:
                if self.periodic_bracket_tax._planner_masks is None:
                    masks = super().generate_masks(completions=completions)
                    self.periodic_bracket_tax._planner_masks = dict(
                        new_taxes=deepcopy(masks[self.world.planner.idx]),
                        zeros={
                            k: np.zeros_like(v)
                            for k, v in masks[self.world.planner.idx].items()
                        },
                    )

                # No need to recompute. Use the cached masks.
                masks = dict()
                if self.periodic_bracket_tax.tax_cycle_pos != 1 or self.disable_taxes:
                    # Apply zero masks for any timestep where taxes are not going to be updated.
                    masks[self.world.planner.idx] = self.periodic_bracket_tax._planner_masks["zeros"]
                else:
                    masks[self.world.planner.idx] = self.periodic_bracket_tax._planner_masks["new_taxes"]

            # Doing annealing.
            else:
                # Figure out what the masks should be this episode.
                if self.periodic_bracket_tax._planner_masks is None:
                    planner_masks = {
                        k: self.periodic_bracket_tax.annealed_tax_mask(
                            completions,
                            self.periodic_bracket_tax._annealing_warmup,
                            self.periodic_bracket_tax._annealing_slope,
                            tax_values,
                        )
                        for k, tax_values in self.periodic_bracket_tax._planner_tax_val_dict.items()
                    }
                    self.periodic_bracket_tax._planner_masks = dict(
                        new_taxes=deepcopy(planner_masks),
                        zeros={k: np.zeros_like(v) for k, v in planner_masks.items()},
                    )

                # No need to recompute. Use the cached masks.
                masks = dict()
                if self.periodic_bracket_tax.tax_cycle_pos != 1 or self.disable_taxes:
                    # Apply zero masks for any timestep where taxes are not going to be updated.
                    masks[self.world.planner.idx] = self.periodic_bracket_tax._planner_masks["zeros"]
                else:
                    masks[self.world.planner.idx] = self.periodic_bracket_tax._planner_masks["new_taxes"]

        # We are not using a learned planner. Generate masks by the default method.
        else:
            masks = super().generate_masks(completions=completions)

        return masks  
    
    def get_metrics(self):
        """
        Metrics that capture what happened through this component.

        Return metrics related to bracket rates, bracket occupancy, and tax collection.
        """
        
        out = dict()

        n_observed_incomes = np.maximum(1, np.sum(list(self.periodic_bracket_tax._occupancy.values())))
        for c in self.periodic_bracket_tax.bracket_cutoffs:
            k = "{:03d}".format(int(c))
            out["avg_bracket_rate/{}".format(k)] = np.mean(self.periodic_bracket_tax._schedules[k])
            out["bracket_occupancy/{}".format(k)] = (
                self.periodic_bracket_tax._occupancy[k] / n_observed_incomes
            )

        if not self.disable_taxes:
            out["avg_effective_tax_rate"] = np.mean(self.periodic_bracket_tax.all_effective_tax_rates)
            out["total_collected_taxes"] = float(self.periodic_bracket_tax.total_collected_taxes)

            # Indices of richest and poorest agents.
            agent_coin_endows = np.array(
                [agent.total_endowment("Coin") for agent in self.world.agents]
            )
            idx_poor = np.argmin(agent_coin_endows)
            idx_rich = np.argmax(agent_coin_endows)

            tax_days = self.periodic_bracket_tax.taxes[(self.period - 1) :: self.period]
            for i, tag in zip([idx_poor, idx_rich], ["poorest", "richest"]):
                total_income = np.maximum(
                    0, [tax_day[str(i)]["income"] for tax_day in tax_days]
                ).sum()
                total_tax_paid = np.sum(
                    [tax_day[str(i)]["tax_paid"] for tax_day in tax_days]
                )
                # Report the overall tax rate over the episode for the richest and poorest agents.
                out["avg_tax_rate/{}".format(tag)] = total_tax_paid / np.maximum(
                    0.001, total_income
                )

            if self.tax_model == "saez":
                # Include the running estimate of elasticity.
                out["saez/estimated_elasticity"] = self.periodic_bracket_tax.elas_tm1

        return out
    
    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Reset tax scheduling trackers and re-sample the planner's vote and invest choices: ["Wood", "Stone", "Iron"].
        """
        
        self.periodic_bracket_tax.curr_rate_indices = [np.random.permutation(21)[0] for _ in range(self.n_brackets)]

        self.periodic_bracket_tax.tax_cycle_pos = 1
        self.periodic_bracket_tax.last_coin = [
            float(agent.total_endowment("Coin")) for agent in self.world.agents
        ]
        self.periodic_bracket_tax.last_income = [0 for _ in range(self.n_agents)]
        self.periodic_bracket_tax.last_marginal_rate = [0 for _ in range(self.n_agents)]
        self.periodic_bracket_tax.last_effective_tax_rate = [0 for _ in range(self.n_agents)]

        self.periodic_bracket_tax._curr_rates_obs = np.array(self.periodic_bracket_tax.curr_marginal_rates)
        self.periodic_bracket_tax._last_income_obs = np.array(self.periodic_bracket_tax.last_income) / self.period
        self.periodic_bracket_tax._last_income_obs_sorted = self.periodic_bracket_tax._last_income_obs[
            np.argsort(self.periodic_bracket_tax._last_income_obs)
        ]

        self.periodic_bracket_tax.taxes = []
        self.periodic_bracket_tax.total_collected_taxes = 0
        self.periodic_bracket_tax.all_effective_tax_rates = []
        self.periodic_bracket_tax._schedules = {"{:03d}".format(int(r)): [] for r in self.periodic_bracket_tax.bracket_cutoffs}
        self.periodic_bracket_tax._occupancy = {"{:03d}".format(int(r)): 0 for r in self.periodic_bracket_tax.bracket_cutoffs}
        self.periodic_bracket_tax._planner_masks = None

        if self.tax_model == "saez":
            self.periodic_bracket_tax.curr_bracket_tax_rates = np.array(self.periodic_bracket_tax.running_avg_tax_rates)
        
        self.votesinvests = []
        
    def get_dense_log(self):
        """
        Log votes and invests and taxes.
        
        Returns:
            taxes (list): A list of tax collections. Each entry corresponds to a single timestep. Entries are empty 
            except for timesteps where a tax period ended and taxes were collected. For those timesteps, each entry
            contains the tax schedule, each agent's reported income, tax paid, and redistribution received.
            
            votesinvests (list): A list of vote and invest events. Each entry corresponds to a single timestep and 
            contains a description of any vote and invest that occurred on that timestep.
        """
        
        if self.disable_taxes:
            return self.votesinvests
        else:
            return self.periodic_bracket_tax.taxes, self.votesinvests