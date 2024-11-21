# Modified by Aslan Satary Dizaji, Copyright (c) 2024.

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy
import numpy as np
from scipy import signal

from modified_ai_economist_gabm.foundation.base.base_env import BaseEnvironment, scenario_registry
from modified_ai_economist_gabm.foundation.scenarios.utils import rewards, social_metrics


@scenario_registry.add
class UniformVoteAndInvest(BaseEnvironment):
    """
    World containing spatially-segregated wood and stone and iron with stochastic regeneration.

    For controlling how resource regeneration behavior...
        Coverage: if fraction, target fraction of total tiles; if integer, target number of tiles.
        Regen Halfwidth: width of regen kernel = 1 + (2 * halfwidth); set >0 to create a spatial social dilemma.
        Regen Weight: regen probability per tile counted by the regen kernel.
        Max Health: how many resource units can populate a source block.
        Clumpiness: degree to which resources are spatially clustered.
        Gradient Steepness: degree to which wood/stone/iron are restricted to the top/bottom of the map.

    Args:
        payment_max_house_building_skill_multiplier (int array): Maximum skill multiplier that an agent
            can sample to build a house (red, blue, green). Must be >= 1. Default is 1.
        planner_gets_spatial_obs (bool): Whether the planner agent receives spatial
            observations from the world.
        full_observability (bool): Whether the mobile agents' spatial observation
            includes the full world view or is instead an egocentric view.
        mobile_agent_observation_range (int): If not using full_observability,
            the spatial range (on each side of the agent) that is visible in the
            spatial observations.
        starting_wood_coverage (int, float): Target coverage of wood at t=0.
        wood_regen_halfwidth (int): Regen halfwidth for wood.
        wood_regen_weight (float): Regen weight for wood.
        wood_max_health (int): Max wood units per wood source tile.
        wood_clumpiness (float): Degree of wood clumping.
        starting_stone_coverage (int, float): Target coverage of stone at t=0.
        stone_regen_halfwidth (int): Regen halfwidth for stone.
        stone_regen_weight (float): Regen weight for stone.
        stone_max_health (int): Max stone units per stone source tile.
        stone_clumpiness (float): Degree of stone clumping.
        starting_iron_coverage (int, float): Target coverage of iron at t=0.
        iron_regen_halfwidth (int): Regen halfwidth for iron.
        iron_regen_weight (float): Regen weight for iron.
        iron_max_health (int): Max iron units per iron source tile.
        iron_clumpiness (float): Degree of iron clumping.
        gradient_steepness (int, float): How steeply source tile probability falls
            off from the top/left of the map.
        checker_source_blocks (bool): Whether to space source tiles in a "checker"
            formation.
        starting_agent_coin (int, float): Amount of coin agents have at t=0. Defaults
            to zero coin.
        isoelastic_eta (float): Parameter controlling the shape of agent utility
            wrt coin endowment.
        energy_cost (float): Coefficient for converting labor to negative utility.
        energy_warmup_constant (float): Decay constant that controls the rate at which
            the effective energy cost is annealed from 0 to energy_cost. Set to 0
            (default) to disable annealing, meaning that the effective energy cost is
            always energy_cost. The units of the decay constant depend on the choice of
            energy_warmup_method.
        energy_warmup_method (str): How to schedule energy annealing (warmup). If
            "decay" (default), use the number of completed episodes. If "auto",
            use the number of timesteps where the average agent reward was positive.
        planner_reward_type (str): The type of reward used for the planner. Options
            are "coin_eq_times_productivity" (default), are "coin_maximin_times_productivity",
            "inv_income_weighted_coin_endowments", and "inv_income_weighted_utility".
        mixing_weight_gini_vs_coin (float): Degree to which equality is ignored w/
            "coin_eq_times_productivity". Default is 0, which weights equality and
            productivity equally. If set to 1, only productivity is rewarded.
        mixing_weight_maximin_vs_coin (float): Degree to which maximin is ignored w/
            "coin_maximin_times_productivity". Default is 0, which weights maximin and
            productivity equally. If set to 1, only productivity is rewarded.
    """

    name = "uniform_scenario_for_vote_and_invest"
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]
    required_entities = ["Wood", "Stone", "Iron", "Coin", "Labor", "VoteInvest", "Expertise"]

    def __init__(
        self,
        *base_env_args,
        payment_max_house_building_skill_multiplier=np.array([15, 5]),
        planner_gets_spatial_info=True,
        full_observability=False,
        mobile_agent_observation_range=5,
        starting_wood_coverage=0.1,
        wood_regen_halfwidth=0.5,
        wood_regen_weight=0.1,
        wood_max_health=1,
        wood_clumpiness=0.35,
        starting_stone_coverage=0.1,
        stone_regen_halfwidth=0.5,
        stone_regen_weight=0.1,
        stone_max_health=1,
        stone_clumpiness=0.35,
        starting_iron_coverage=0.1,
        iron_regen_halfwidth=0.5,
        iron_regen_weight=0.1,
        iron_max_health=1,
        iron_clumpiness=0.35,
        gradient_steepness=1,
        checker_source_blocks=False,
        starting_agent_coin=50,
        isoelastic_eta=0.23,
        energy_cost=0.21,
        energy_warmup_constant=0,
        energy_warmup_method='decay',
        planner_reward_type='coin_maximin_times_productivity',
        mixing_weight_gini_vs_coin=0.0,
        mixing_weight_maximin_vs_coin=0.0,
        **base_env_kwargs
    ):
        
        super().__init__(*base_env_args, **base_env_kwargs)

        self.payment_max_house_building_skill_multiplier = payment_max_house_building_skill_multiplier
        assert np.all(self.payment_max_house_building_skill_multiplier >= 1)

        # Whether agents receive spatial information in their observation tensor
        self._planner_gets_spatial_info = bool(planner_gets_spatial_info)

        # Whether the (non-planner) agents can see the whole world map
        self._full_observability = bool(full_observability)

        self._mobile_agent_observation_range = int(mobile_agent_observation_range)

        # For controlling how resource regeneration behavior:
        #  - Coverage: if fraction, target fraction of total tiles; if integer, target number of tiles.
        #  - Regen Halfwidth: width of regen kernel = 1 + (2 * halfwidth).
        #  - Regen Weight: regen probability per tile counted by the regen kernel.
        #  - Max Health: how many resource units can populate a source block.
        #  - Clumpiness: degree to which resources are spatially clustered.
        #  - Gradient Steepness: degree to which wood/stone/iron are restricted to top/left of map.
        
        self.layout_specs = dict(Wood={}, Stone={}, Iron={}, Soil={})
        
        if starting_wood_coverage >= 1:
            starting_wood_coverage /= np.prod(self.world_size)
        if starting_stone_coverage >= 1:
            starting_stone_coverage /= np.prod(self.world_size)
        if starting_iron_coverage >= 1:
            starting_iron_coverage /= np.prod(self.world_size)
        assert (starting_wood_coverage + starting_stone_coverage + starting_iron_coverage) < 0.75
        #
        self._checker_source_blocks = bool(checker_source_blocks)
        c, r = np.meshgrid(
            np.arange(self.world_size[1]) % 2, np.arange(self.world_size[0]) % 2
        )
        self._checker_mask = (r + c) == 1
        m = 2 if self._checker_source_blocks else 1
        #
        self.layout_specs["Wood"]["starting_coverage"] = (
            float(starting_wood_coverage) * m
        )
        self.layout_specs["Stone"]["starting_coverage"] = (
            float(starting_stone_coverage) * m
        )
        self.layout_specs["Iron"]["starting_coverage"] = (
            float(starting_iron_coverage) * m
        )
        assert 0 < self.layout_specs["Wood"]["starting_coverage"] < 1
        assert 0 < self.layout_specs["Stone"]["starting_coverage"] < 1
        assert 0 < self.layout_specs["Iron"]["starting_coverage"] < 1
        #
        self.layout_specs["Wood"]["regen_halfwidth"] = int(wood_regen_halfwidth)
        self.layout_specs["Stone"]["regen_halfwidth"] = int(stone_regen_halfwidth)
        self.layout_specs["Iron"]["regen_halfwidth"] = int(iron_regen_halfwidth)
        assert 0 <= self.layout_specs["Wood"]["regen_halfwidth"] <= 3
        assert 0 <= self.layout_specs["Stone"]["regen_halfwidth"] <= 3
        assert 0 <= self.layout_specs["Iron"]["regen_halfwidth"] <= 3
        #
        self.layout_specs["Wood"]["regen_weight"] = float(wood_regen_weight)
        self.layout_specs["Stone"]["regen_weight"] = float(stone_regen_weight)
        self.layout_specs["Iron"]["regen_weight"] = float(iron_regen_weight)
        assert 0 <= self.layout_specs["Wood"]["regen_weight"] <= 1
        assert 0 <= self.layout_specs["Stone"]["regen_weight"] <= 1
        assert 0 <= self.layout_specs["Iron"]["regen_weight"] <= 1
        #
        self.layout_specs["Wood"]["max_health"] = int(wood_max_health)
        self.layout_specs["Stone"]["max_health"] = int(stone_max_health)
        self.layout_specs["Iron"]["max_health"] = int(iron_max_health)
        assert self.layout_specs["Wood"]["max_health"] > 0
        assert self.layout_specs["Stone"]["max_health"] > 0
        assert self.layout_specs["Iron"]["max_health"] > 0
        #
        self.clumpiness = {
            "Wood": float(wood_clumpiness),
            "Stone": float(stone_clumpiness),
            "Iron": float(iron_clumpiness),
        }
        assert all(0 <= v <= 1 for v in self.clumpiness.values())
        #
        self.gradient_steepness = float(gradient_steepness)
        assert self.gradient_steepness >= 1.0
        #
        self.source_prob_maps = self.make_source_prob_maps()
        self.source_maps = {
            k: np.zeros_like(v) for k, v in self.source_prob_maps.items()
        }

        # How much coin do agents begin with at upon reset
        self.starting_agent_coin = float(starting_agent_coin)
        assert self.starting_agent_coin >= 0.0

        # Controls the diminishing marginal utility of coin.
        # isoelastic_eta=0 means no diminishing utility.
        self.isoelastic_eta = float(isoelastic_eta)
        assert 0.0 <= self.isoelastic_eta <= 1.0

        # The amount that labor is weighted in utility computation
        # (once annealing is finished)
        self.energy_cost = float(energy_cost)
        assert self.energy_cost >= 0

        # What value to use for calculating the progress of energy annealing
        # If method = 'decay': #completed episodes
        # If method = 'auto' : #timesteps where avg. agent reward > 0
        self.energy_warmup_method = energy_warmup_method.lower()
        assert self.energy_warmup_method in ["decay", "auto"]
        # Decay constant for annealing to full energy cost
        # (if energy_warmup_constant == 0, there is no annealing)
        self.energy_warmup_constant = float(energy_warmup_constant)
        assert self.energy_warmup_constant >= 0
        self._auto_warmup_integrator = 0

        # Which social welfare function to use
        self.planner_reward_type = str(planner_reward_type).lower()

        # How much to weight equality if using SWF=eq*prod:
        # 0 -> SWF=eq*prod
        # 1 -> SWF=prod
        
        self.mixing_weight_gini_vs_coin = float(mixing_weight_gini_vs_coin)
        assert 0 <= self.mixing_weight_gini_vs_coin <= 1.0
        
        self.mixing_weight_maximin_vs_coin = float(mixing_weight_maximin_vs_coin)
        assert 0 <= self.mixing_weight_maximin_vs_coin <= 1.0

        # Use this to calculate marginal changes and deliver that as reward
        self.init_optimization_metric = {agent.idx: 0 for agent in self.all_agents}
        self.prev_optimization_metric = {agent.idx: 0 for agent in self.all_agents}
        self.curr_optimization_metric = {agent.idx: 0 for agent in self.all_agents}

    @property
    def energy_weight(self):
        """
        Energy annealing progress. Multiply with self.energy_cost to get the
        effective energy coefficient.
        """
        
        if self.energy_warmup_constant <= 0.0:
            return 1.0

        if self.energy_warmup_method == "decay":
            return float(1.0 - np.exp(-self._completions / self.energy_warmup_constant))

        if self.energy_warmup_method == "auto":
            return float(
                1.0
                - np.exp(-self._auto_warmup_integrator / self.energy_warmup_constant)
            )

        raise NotImplementedError

    def get_current_optimization_metrics(self):
        """
        Compute optimization metrics based on the current state. Used to compute reward.

        Returns:
            curr_optimization_metric (dict): A dictionary of {agent.idx: metric}
                with an entry for each agent (including the planner) in the env.
        """
        
        curr_optimization_metric = {}
        
        # (for agents)
        for agent in self.world.agents:
            curr_optimization_metric[agent.idx] = rewards.isoelastic_coin_minus_labor(
                coin_endowment=agent.total_endowment("Coin"),
                total_labor=agent.state["endogenous"]["Labor"],
                isoelastic_eta=self.isoelastic_eta,
                labor_coefficient=self.energy_weight * self.energy_cost,
            )
        
        # (for the planner)
        if self.planner_reward_type == "coin_eq_times_productivity":
            curr_optimization_metric[
                self.world.planner.idx
            ] = rewards.coin_eq_times_productivity(
                coin_endowments=np.array(
                    [agent.total_endowment("Coin") for agent in self.world.agents]
                ),
                equality_weight=1 - self.mixing_weight_gini_vs_coin,
            )
        
        elif self.planner_reward_type == "coin_maximin_times_productivity":
            curr_optimization_metric[
                self.world.planner.idx
            ] = rewards.coin_maximin_times_productivity(
                coin_endowments=np.array(
                    [agent.total_endowment("Coin") for agent in self.world.agents]
                ),
                maximin_weight=1 - self.mixing_weight_maximin_vs_coin,
            )
       
        elif self.planner_reward_type == "inv_income_weighted_coin_endowments":
            curr_optimization_metric[
                self.world.planner.idx
            ] = rewards.inv_income_weighted_coin_endowments(
                coin_endowments=np.array(
                    [agent.total_endowment("Coin") for agent in self.world.agents]
                )
            )
        
        elif self.planner_reward_type == "inv_income_weighted_utility":
            curr_optimization_metric[
                self.world.planner.idx
            ] = rewards.inv_income_weighted_utility(
                coin_endowments=np.array(
                    [agent.total_endowment("Coin") for agent in self.world.agents]
                ),
                utilities=np.array(
                    [curr_optimization_metric[agent.idx] for agent in self.world.agents]
                ),
            )
        
        else:
            print("No valid planner reward selected!")
            raise NotImplementedError
        
        return curr_optimization_metric

    def make_source_prob_maps(self):
        """
        Make maps specifying how likely each location is to be assigned as a resource
        source tile.

        Returns:
            source_prob_maps (dict): Contains a source probability map for wood and stone and iron.
        """
        
        prob_gradient = (
            np.arange(self.world_size[0])[:, None].repeat(self.world_size[1], axis=1)
            ** self.gradient_steepness
        )
        prob_gradient = prob_gradient / np.mean(prob_gradient)

        return {
            "Wood": prob_gradient[-1::-1, :] * self.layout_specs["Wood"]["starting_coverage"],
            "Stone": prob_gradient[:, -1::-1] * self.layout_specs["Stone"]["starting_coverage"],
            "Iron": prob_gradient[-1::-1] * self.layout_specs["Iron"]["starting_coverage"],
        }

    # The following methods must be implemented for each scenario
    # -----------------------------------------------------------

    def reset_starting_layout(self):
        """
        Part 1/2 of scenario reset. This method handles resetting the state of the
        environment managed by the scenario (i.e. resource & landmark layout).

        Here, generate a resource source layout consistent with target parameters.
        """
        
        happy_coverage = False
        n_reset_tries = 0

        # Attempt to do a reset until an attempt limit is reached or coverage is good
        while n_reset_tries < 100 and not happy_coverage:
            self.world.maps.clear()

            self.source_maps = {
                k: np.zeros_like(v) for k, v in self.source_prob_maps.items()
            }

            resources = ["Wood", "Stone", "Iron"]

            for resource in resources:
                clump = 1 - np.clip(self.clumpiness[resource], 0.0, 0.99)

                source_prob = self.source_prob_maps[resource] * 0.1 * clump

                empty = self.world.maps.empty

                tmp = np.random.rand(*source_prob.shape)
                maybe_source_map = (tmp < source_prob) * empty

                n_tries = 0
                while np.mean(maybe_source_map) < (
                    self.layout_specs[resource]["starting_coverage"] * clump
                ):
                    tmp *= 0.9
                    maybe_source_map = (tmp < source_prob) * empty
                    n_tries += 1
                    if n_tries > 200:
                        break

                while (
                    np.mean(maybe_source_map)
                    < self.layout_specs[resource]["starting_coverage"]
                ):
                    kernel = np.random.randn(7, 7) > 0
                    tmp = signal.convolve2d(
                        maybe_source_map
                        + (0.2 * np.random.randn(*maybe_source_map.shape))
                        - 0.25,
                        kernel.astype(np.float32),
                        "same",
                    )
                    maybe_source_map = np.maximum(tmp > 0, maybe_source_map) * empty

                self.source_maps[resource] = maybe_source_map
                self.world.maps.set(
                    resource, maybe_source_map
                )  # * self.layout_specs[resource]['max_health'])
                self.world.maps.set(resource + "SourceBlock", maybe_source_map)

            # Restart if the resource distribution is too far off the target coverage
            happy_coverage = True
            for resource in resources:
                coverage_quotient = (
                    np.mean(self.source_maps[resource])
                    / self.layout_specs[resource]["starting_coverage"]
                )
                bound = 0.4
                if not (1 / (1 + bound)) <= coverage_quotient <= (1 + bound):
                    happy_coverage = False

            n_reset_tries += 1

        # Apply checkering, if applicable
        if self._checker_source_blocks:
            for resource, source_map in self.source_maps.items():
                source_map = source_map * self._checker_mask
                self.source_maps[resource] = source_map
                self.world.maps.set(resource, source_map)
                self.world.maps.set(resource + "SourceBlock", source_map)

    def reset_agent_states(self):
        """
        Part 2/2 of scenario reset. This method handles resetting the state of the
        agents themselves (i.e. inventory, locations, etc.).

        Here, empty inventories, give mobile agents any starting coin, and place them
        in random accessible locations to start.
        """
        
        self.world.clear_agent_locs()

        # Clear everything to start with the starting agent coins
        for agent in self.world.agents:
            agent.state["inventory"]["Coin"] = float(self.starting_agent_coin)
            agent.state["escrow"]["Coin"] = 0.0
        
            agent.state["inventory"]["Wood"] = int(self.starting_agent_coin)
            agent.state["escrow"]["Wood"] = 0
            agent.state["inventory"]["Stone"] = int(self.starting_agent_coin)
            agent.state["escrow"]["Stone"] = 0
            agent.state["inventory"]["Iron"] = int(self.starting_agent_coin)
            agent.state["escrow"]["Iron"] = 0
            
            agent.state["endogenous"]["Labor"] = 0.0 
            agent.state["endogenous"]["VoteInvest"] = float(self.starting_agent_coin) * np.random.random(3)
            
            if float(agent.idx) / 2 == float(agent.idx) // 2:
                agent.state["endogenous"]["Expertise"] = 'Expert'
                agent.state["payment_max_house_building_skill_multiplier"] = self.payment_max_house_building_skill_multiplier[0]
            else:
                agent.state["endogenous"]["Expertise"] = 'Novice'
                agent.state["payment_max_house_building_skill_multiplier"] = self.payment_max_house_building_skill_multiplier[1]

        # Clear everything for the planner
        self.world.planner.state["inventory"] = {
            k: 0 for k in self.world.planner.inventory.keys()
        }
        self.world.planner.state["escrow"] = {
            k: 0 for k in self.world.planner.escrow.keys()
        }
        self.world.planner.state["endogenous"]["VoteInvest"] = float(self.starting_agent_coin) * np.random.random(3)
        self.world.planner.state["endogenous"]["Expertise"] = []

        # Place the agents randomly in the world
        for agent in self.world.get_random_order_agents():
            r = np.random.randint(0, self.world_size[0])
            c = np.random.randint(0, self.world_size[1])
            n_tries = 0
            while not self.world.can_agent_occupy(r, c, agent):
                r = np.random.randint(0, self.world_size[0])
                c = np.random.randint(0, self.world_size[1])
                n_tries += 1
                if n_tries > 200:
                    raise TimeoutError
            self.world.set_agent_loc(agent, r, c)

    def scenario_step(self, investments):
        """
        Update the state of the world according to whatever rules this scenario
        implements.

        This gets called in the 'step' method (of base_env) after going through each
        component step and before generating observations, rewards, etc.

        In this class of scenarios, the scenario step handles stochastic resource
        regeneration.
        """

        resources = ["Wood", "Stone", "Iron"]
        minimum_investment = 0.0
        counter = -1

        for resource in resources:
            counter = counter + 1
            
            d = 1 + (2 * self.layout_specs[resource]["regen_halfwidth"] * int(minimum_investment + investments[counter]))
            
            kernel = (
                self.layout_specs[resource]["regen_weight"] * (minimum_investment + investments[counter]) * np.ones((d, d)) / (d ** 2)
            )

            resource_map = self.world.maps.get(resource)
            resource_source_blocks = self.world.maps.get(resource + "SourceBlock")
            spawnable = (
                self.world.maps.empty + resource_map + resource_source_blocks
            ) > 0
            spawnable *= resource_source_blocks > 0

            health = np.maximum(resource_map, resource_source_blocks)
            respawn = np.random.rand(*health.shape) < signal.convolve2d(
                health, kernel, "same"
            )
            respawn *= spawnable

            self.world.maps.set(
                resource,
                np.minimum(
                    resource_map + respawn, self.layout_specs[resource]["max_health"] * (minimum_investment + investments[counter])
                ),
            )

    def generate_observations(self):
        """
        Generate observations associated with this scenario.

        A scenario does not need to produce observations and can provide observations
        for only some agent types; however, for a given agent type, it should either
        always or never yield an observation. If it does yield an observation,
        that observation should always have the same structure/sizes!

        Returns:
            obs (dict): A dictionary of {agent.idx: agent_obs_dict}. In words,
                return a dictionary with an entry for each agent (which can including
                the planner) for which this scenario provides an observation. For each
                entry, the key specifies the index of the agent and the value contains
                its associated observation dictionary.

        Here, non-planner agents receive spatial observations (depending on the env
        config) as well as the contents of their inventory and endogenous quantities.
        The planner also receives spatial observations (again, depending on the env
        config) as well as the inventory of each of the mobile agents.
        """
        
        obs = {}
        curr_map = self.world.maps.state

        owner_map = self.world.maps.owner_state
        loc_map = self.world.loc_map
        agent_idx_maps = np.concatenate([owner_map, loc_map[None, :, :]], axis=0)
        agent_idx_maps += 2
        agent_idx_maps[agent_idx_maps == 1] = 0

        agent_locs = {
            str(agent.idx): {
                "loc-row": agent.loc[0] / self.world_size[0],
                "loc-col": agent.loc[1] / self.world_size[1],
            }
            for agent in self.world.agents
        }
        agent_invs = {
            str(agent.idx): {
                "inventory-" + k: v * self.inv_scale for k, v in agent.inventory.items()
            }
            for agent in self.world.agents
        }

        obs[self.world.planner.idx] = {
            "inventory-" + k: v * self.inv_scale
            for k, v in self.world.planner.inventory.items()
        }
        if self._planner_gets_spatial_info:
            obs[self.world.planner.idx].update(
                dict(map=curr_map, idx_map=agent_idx_maps)
            )
        
        # Mobile agents see the full map. Convey location info via one-hot map channels.
        if self._full_observability:
            for agent in self.world.agents:
                my_map = np.array(agent_idx_maps)
                my_map[my_map == int(agent.idx) + 2] = 1
                sidx = str(agent.idx)
                obs[sidx] = {"map": curr_map, "idx_map": my_map}
                obs[sidx].update(agent_invs[sidx])

        # Mobile agents only see within a window around their position
        else:
            w = (
                self._mobile_agent_observation_range
            )  # View halfwidth (only applicable without full observability)

            padded_map = np.pad(
                curr_map,
                [(0, 1), (w, w), (w, w)],
                mode="constant",
                constant_values=[(0, 1), (0, 0), (0, 0)],
            )

            padded_idx = np.pad(
                agent_idx_maps,
                [(0, 0), (w, w), (w, w)],
                mode="constant",
                constant_values=[(0, 0), (0, 0), (0, 0)],
            )

            for agent in self.world.agents:
                r, c = [c + w for c in agent.loc]
                visible_map = padded_map[
                    :, (r - w) : (r + w + 1), (c - w) : (c + w + 1)
                ]
                visible_idx = np.array(
                    padded_idx[:, (r - w) : (r + w + 1), (c - w) : (c + w + 1)]
                )

                visible_idx[visible_idx == int(agent.idx) + 2] = 1

                sidx = str(agent.idx)

                obs[sidx] = {"map": visible_map, "idx_map": visible_idx}
                obs[sidx].update(agent_locs[sidx])
                obs[sidx].update(agent_invs[sidx])

                # Agent-wise planner info (gets crunched into the planner obs in the base scenario code)
                obs["p" + sidx] = agent_invs[sidx]
                if self._planner_gets_spatial_info:
                    obs["p" + sidx].update(agent_locs[sidx])

        return obs

    def compute_reward(self):
        """
        Apply the reward function(s) associated with this scenario to get the rewards
        from this step.

        Returns:
            rew (dict): A dictionary of {agent.idx: agent_obs_dict}. In words,
                return a dictionary with an entry for each agent in the environment
                (including the planner). For each entry, the key specifies the index of
                the agent and the value contains the scalar reward earned this timestep.

        Rewards are computed as the marginal utility (agents) or marginal social
        welfare (planner) experienced on this timestep. Ignoring discounting,
        this means that agents' (planner's) objective is to maximize the utility
        (social welfare) associated with the terminal state of the episode.
        """

        # "curr_optimization_metric" hasn't been updated yet, so it gives us the utility from the last step.
        utility_at_end_of_last_time_step = deepcopy(self.curr_optimization_metric)

        # compute current objectives and store the values
        self.curr_optimization_metric = self.get_current_optimization_metrics()

        # reward = curr - prev objectives
        rew = {
            k: float(v - utility_at_end_of_last_time_step[k])
            for k, v in self.curr_optimization_metric.items()
        }

        # store the previous objective values
        self.prev_optimization_metric.update(utility_at_end_of_last_time_step)

        # Automatic Energy Cost Annealing
        # -------------------------------
        avg_agent_rew = np.mean([rew[a.idx] for a in self.world.agents])
        # Count the number of timesteps where the avg agent reward was > 0
        if avg_agent_rew > 0:
            self._auto_warmup_integrator += 1

        return rew

    # Optional methods for customization
    # ----------------------------------

    def additional_reset_steps(self):
        """
        Extra scenario-specific steps that should be performed at the end of the reset
        cycle.

        For each reset cycle...
            First, reset_starting_layout() and reset_agent_states() will be called.

            Second, <component>.reset() will be called for each registered component.

            Lastly, this method will be called to allow for any final customization of
            the reset cycle.

        For this scenario, this method resets optimization metric trackers.
        """
        
        # compute current objectives
        curr_optimization_metric = self.get_current_optimization_metrics()

        self.curr_optimization_metric = deepcopy(curr_optimization_metric)
        self.init_optimization_metric = deepcopy(curr_optimization_metric)
        self.prev_optimization_metric = deepcopy(curr_optimization_metric)

    def scenario_metrics(self):
        """
        Allows the scenario to generate metrics (collected along with component metrics
        in the 'metrics' property).

        To have the scenario add metrics, this function needs to return a dictionary of
        {metric_key: value} where 'value' is a scalar (no nesting or lists!)

        Here, summarize social metrics, endowments, utilities, and labor cost annealing
        """
        
        metrics = dict()

        coin_endowments = np.array(
            [agent.total_endowment("Coin") for agent in self.world.agents]
        )
        
        metrics["social/productivity"] = social_metrics.get_productivity(coin_endowments)        
        metrics["social/equality"] = social_metrics.get_equality(coin_endowments)
        metrics["social/maximin"] = social_metrics.get_maximin(coin_endowments)

        
        utilities = np.array(
            [self.curr_optimization_metric[agent.idx] for agent in self.world.agents]
        )
        
        metrics[
            "social_welfare/coin_eq_times_productivity"
        ] = rewards.coin_eq_times_productivity(
            coin_endowments=coin_endowments, equality_weight=1.0
        )
        
        metrics[
            "social_welfare/coin_maximin_times_productivity"
        ] = rewards.coin_maximin_times_productivity(
            coin_endowments=coin_endowments, maximin_weight=1.0
        )
        
        metrics[
            "social_welfare/inv_income_weighted_coin_endow"
        ] = rewards.inv_income_weighted_coin_endowments(coin_endowments=coin_endowments)
        
        metrics[
            "social_welfare/inv_income_weighted_utility"
        ] = rewards.inv_income_weighted_utility(
            coin_endowments=coin_endowments, utilities=utilities
        )

        for agent in self.all_agents:
            for resource, quantity in agent.inventory.items():
                metrics[
                    "endow/{}/{}".format(agent.idx, resource)
                ] = agent.total_endowment(resource)

            if agent.endogenous is not None:
                for resource, quantity in agent.endogenous.items():
                    metrics["endogenous/{}/{}".format(agent.idx, resource)] = quantity

            metrics["util/{}".format(agent.idx)] = self.curr_optimization_metric[
                agent.idx
            ]

        # Labor weight
        metrics["labor/weighted_cost"] = self.energy_cost * self.energy_weight
        metrics["labor/warmup_integrator"] = int(self._auto_warmup_integrator)

        return metrics