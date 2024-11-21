# Modified by Aslan Satary Dizaji, Copyright (c) 2024.

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from modified_ai_economist_gabm.foundation.base.base_component import BaseComponent, component_registry


@component_registry.add
class build_and_trade_house_vs_skill_component(BaseComponent):
    """Mobile agents are divided to two categories depending on their house building skill, expert and novice. 
       Expert mobile agents have high initial house building skill for all three types of houses.
       Novice mobile agents have low initial house building skill for all three types of houses. 
       Expert mobile agents can build three types of houses and earn money, or can sell three types of houses and earn money, 
       or can sell their house building skill and simultaneously earn money and increase their house building skill.
       Novice mobile agents can build three types of houses if their house building skill is higher than a minimum and earn money, 
       can buy three types of houses and earn money, or can buy house building skill and simultaneously earn money and increase their house building skill.

    Can be configured to include heterogeneous building skill/labor where agents earn different levels of 
    income or tolerate different levels of labor when building houses. 

    Args:
        payment (int): Default amount of coin agents earn from building.
            Must be >= 0. Default is 10.
        payment_max_skill_multiplier (int array): Maximum skill multiplier that an agent
            can sample. Must be >= 1. Default is 1.
        skill_dist (str): Distribution type for sampling skills. Default ("none")
            gives all agents identical skill equal to a multiplier of 1. "pareto" and
            "lognormal" sample skills from the associated distributions.
        build_labor (float array): Labor cost associated with building a house.
            Must be >= 0. Default is 10.
    """

    name = "BuildTradeHouseSkill"
    component_type = "BuildTradeHouseSkill"
    required_entities = ["Wood", "Stone", "Iron", "Coin", "RedHouse", "BlueHouse", "GreenHouse", "Labor", "Expertise"]
    agent_subclasses = ["BasicMobileAgent"]
    
    def __init__(
        self,
        *base_component_args,
        payment=10,
        payment_max_skill_multiplier=np.array([15, 5]),
        skill_dist="pareto",
        build_labor=10,
        **base_component_kwargs
    ):
        
        super().__init__(*base_component_args, **base_component_kwargs)

        self.payment = int(payment)
        assert self.payment >= 0

        self.payment_max_skill_multiplier = payment_max_skill_multiplier
        assert np.all(self.payment_max_skill_multiplier >= 1)

        self.resource_cost = {"Wood": 1, "Stone": 1, "Iron": 1}

        self.build_labor = build_labor
        assert np.all(self.build_labor >= 0.0)

        self.skill_dist = skill_dist.lower()
        assert self.skill_dist in ["none", "pareto", "lognormal"]

        self.sampled_skills_red_house = {}
        self.sampled_skills_blue_house = {}
        self.sampled_skills_green_house = {}

        self.builds = []

    def agent_can_build(self, agent):
        """Return True if agent can actually build in its current location."""
        
        # See if the agent has the resources necessary to complete the action
        for resource, cost in self.resource_cost.items():
            if agent.state["inventory"][resource] < cost:
                return False

        # Do nothing if this spot is already occupied by a landmark or resource
        if self.world.location_resources(*agent.loc):
            return False
        if self.world.location_landmarks(*agent.loc):
            return False
        # If we made it here, the agent can build.
        return True

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        Add three actions (building three different houses) for mobile agents.
        """
        
        # This component adds three actions that mobile agents can take: building three different houses.
        if agent_cls_name == "BasicMobileAgent":
            return 7

        return None

    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        For mobile agents, add state fields for building skill.
        """
        
        if agent_cls_name not in self.agent_subclasses:
            return {}
        if agent_cls_name == "BasicMobileAgent":
            return {"build_payment_red_house": float(self.payment), "build_payment_blue_house": float(self.payment), "build_payment_green_house": float(self.payment),
                    "build_skill_red_house": 1, "build_skill_blue_house": 1, "build_skill_green_house": 1}
        
        raise NotImplementedError

    def component_step(self):
        """
        See base_component.py for detailed description.

        Convert stone+wood+iron to house+coin for agents that choose to build and can, 
        or sell or buy house or skill for agents that choose and can.
        """
         
        build = []

        agents_counted_action_trade_red_house = []
        agents_counted_action_trade_blue_house = []
        agents_counted_action_trade_green_house = []

        agents_counted_action_trade_build_skill = []

        random_order_agents_actions = []
        random_order_agents_expertise = []

        random_order_agents = self.world.get_random_order_agents()

        for agent in random_order_agents:
            random_order_agents_actions.append(agent.get_component_action(self.name))
            random_order_agents_expertise.append(agent.state["endogenous"]["Expertise"])

        # Apply any building actions taken by the mobile agents
        for i in range(len(random_order_agents)):

            # This component doesn't apply to this agent!
            if random_order_agents_actions[i] is None:
                continue

            # NO-OP!
            if random_order_agents_actions[i] == 0:
                pass

            elif random_order_agents_actions[i] == 1 and random_order_agents_expertise[i] == "Expert":
                # Build a red house if you can!
                self.resource_cost = {"Wood": 1, "Stone": 1}
                
                if self.agent_can_build(random_order_agents[i]):
                    # Remove the resources
                    for resource, cost in self.resource_cost.items():
                        random_order_agents[i].state["inventory"][resource] -= cost

                    # Place a house where the agent is standing
                    loc_r, loc_c = random_order_agents[i].loc
                    self.world.create_landmark("RedHouse", loc_r, loc_c, random_order_agents[i].idx)

                    # Receive payment for the house
                    random_order_agents[i].state["inventory"]["Coin"] += random_order_agents[i].state["build_payment_red_house"]

                    # Incur the labor cost for building
                    random_order_agents[i].state["endogenous"]["Labor"] += self.build_labor

                    build.append(
                        {
                            "builder": random_order_agents[i].idx,
                            "type": "red_house",
                            "loc": np.array(random_order_agents[i].loc),
                            "income": float(random_order_agents[i].state["build_payment_red_house"]),
                        }
                    )

            elif random_order_agents_actions[i] == 1 and random_order_agents_expertise[i] == "Novice":
                # Build a red house if you can!
                self.resource_cost = {"Wood": 1, "Stone": 1}
                
                if self.agent_can_build(random_order_agents[i]) and random_order_agents[i].state["payment_max_house_building_skill_multiplier"] >= 15:
                    # Remove the resources
                    for resource, cost in self.resource_cost.items():
                        random_order_agents[i].state["inventory"][resource] -= cost

                    # Place a house where the agent is standing
                    loc_r, loc_c = random_order_agents[i].loc
                    self.world.create_landmark("RedHouse", loc_r, loc_c, random_order_agents[i].idx)

                    # Receive payment for the house
                    random_order_agents[i].state["inventory"]["Coin"] += random_order_agents[i].state["build_payment_red_house"]

                    # Incur the labor cost for building
                    random_order_agents[i].state["endogenous"]["Labor"] += self.build_labor

                    build.append(
                        {
                            "builder": random_order_agents[i].idx,
                            "type": "red_house",
                            "loc": np.array(random_order_agents[i].loc),
                            "income": float(random_order_agents[i].state["build_payment_red_house"]),
                        }
                    )

            elif random_order_agents_actions[i] == 2 and random_order_agents_expertise[i] == "Expert":
                # Build a blue house if you can!
                self.resource_cost = {"Wood": 1, "Iron": 1}
                
                if self.agent_can_build(random_order_agents[i]):
                    # Remove the resources
                    for resource, cost in self.resource_cost.items():
                        random_order_agents[i].state["inventory"][resource] -= cost

                    # Place a house where the agent is standing
                    loc_r, loc_c = random_order_agents[i].loc
                    self.world.create_landmark("BlueHouse", loc_r, loc_c, random_order_agents[i].idx)

                    # Receive payment for the house
                    random_order_agents[i].state["inventory"]["Coin"] += random_order_agents[i].state["build_payment_blue_house"]

                    # Incur the labor cost for building
                    random_order_agents[i].state["endogenous"]["Labor"] += self.build_labor

                    build.append(
                        {
                            "builder": random_order_agents[i].idx,
                            "type": "blue_house",
                            "loc": np.array(random_order_agents[i].loc),
                            "income": float(random_order_agents[i].state["build_payment_blue_house"]),
                        }
                    )

            elif random_order_agents_actions[i] == 2 and random_order_agents_expertise[i] == "Novice":
                # Build a blue house if you can!
                self.resource_cost = {"Wood": 1, "Iron": 1}
                
                if self.agent_can_build(random_order_agents[i]) and random_order_agents[i].state["payment_max_house_building_skill_multiplier"] >= 15:
                    # Remove the resources
                    for resource, cost in self.resource_cost.items():
                        random_order_agents[i].state["inventory"][resource] -= cost

                    # Place a house where the agent is standing
                    loc_r, loc_c = random_order_agents[i].loc
                    self.world.create_landmark("BlueHouse", loc_r, loc_c, random_order_agents[i].idx)

                    # Receive payment for the house
                    random_order_agents[i].state["inventory"]["Coin"] += random_order_agents[i].state["build_payment_blue_house"]

                    # Incur the labor cost for building
                    random_order_agents[i].state["endogenous"]["Labor"] += self.build_labor

                    build.append(
                        {
                            "builder": random_order_agents[i].idx,
                            "type": "blue_house",
                            "loc": np.array(random_order_agents[i].loc),
                            "income": float(random_order_agents[i].state["build_payment_blue_house"]),
                        }
                    )

            elif random_order_agents_actions[i] == 3 and random_order_agents_expertise[i] == "Expert":
                # Build a green house if you can!
                self.resource_cost = {"Stone": 1, "Iron": 1}
                
                if self.agent_can_build(random_order_agents[i]):
                    # Remove the resources
                    for resource, cost in self.resource_cost.items():
                        random_order_agents[i].state["inventory"][resource] -= cost

                    # Place a house where the agent is standing
                    loc_r, loc_c = random_order_agents[i].loc
                    self.world.create_landmark("GreenHouse", loc_r, loc_c, random_order_agents[i].idx)

                    # Receive payment for the house
                    random_order_agents[i].state["inventory"]["Coin"] += random_order_agents[i].state["build_payment_green_house"]

                    # Incur the labor cost for building
                    random_order_agents[i].state["endogenous"]["Labor"] += self.build_labor

                    build.append(
                        {
                            "builder": random_order_agents[i].idx,
                            "type": "green_house",
                            "loc": np.array(random_order_agents[i].loc),
                            "income": float(random_order_agents[i].state["build_payment_green_house"]),
                        }
                    )

            elif random_order_agents_actions[i] == 3 and random_order_agents_expertise[i] == "Novice":
                # Build a green house if you can!
                self.resource_cost = {"Stone": 1, "Iron": 1}
                
                if self.agent_can_build(random_order_agents[i]) and random_order_agents[i].state["payment_max_house_building_skill_multiplier"] >= 15:
                    # Remove the resources
                    for resource, cost in self.resource_cost.items():
                        random_order_agents[i].state["inventory"][resource] -= cost

                    # Place a house where the agent is standing
                    loc_r, loc_c = random_order_agents[i].loc
                    self.world.create_landmark("GreenHouse", loc_r, loc_c, random_order_agents[i].idx)

                    # Receive payment for the house
                    random_order_agents[i].state["inventory"]["Coin"] += random_order_agents[i].state["build_payment_green_house"]

                    # Incur the labor cost for building
                    random_order_agents[i].state["endogenous"]["Labor"] += self.build_labor

                    build.append(
                        {
                            "builder": random_order_agents[i].idx,
                            "type": "green_house",
                            "loc": np.array(random_order_agents[i].loc),
                            "income": float(random_order_agents[i].state["build_payment_green_house"]),
                        }
                    )

            # Sell/Buy a red house if it is possible!
            elif random_order_agents_actions[i] == 4 and random_order_agents_expertise[i] == "Expert":
                agents_counted_action_trade_red_house.append(random_order_agents[i])

                # Apply any building actions taken by the mobile agents
                for j in range(len(random_order_agents_actions)):

                    if random_order_agents_actions[j] == 4 and random_order_agents_expertise[j] == "Novice" and random_order_agents[j] not in agents_counted_action_trade_red_house:

                        agents_counted_action_trade_red_house.append(random_order_agents[j])

                        # Build and sell/buy a red house if you can!
                        self.resource_cost = {"Wood": 1, "Stone": 1}
                
                        if self.agent_can_build(random_order_agents[i]):
                            for resource, cost in self.resource_cost.items():
                                random_order_agents[i].state["inventory"][resource] -= cost

                            # Place a house where the agent is standing
                            loc_r, loc_c = random_order_agents[i].loc
                            self.world.create_landmark("RedHouse", loc_r, loc_c, random_order_agents[i].idx)

                            # Receive payment for the house
                            random_order_agents[i].state["inventory"]["Coin"] += random_order_agents[i].state["build_payment_red_house"]
                            random_order_agents[j].state["inventory"]["Coin"] += random_order_agents[j].state["build_payment_red_house"]

                            # Incur the labor cost for building
                            random_order_agents[i].state["endogenous"]["Labor"] += 0
                            random_order_agents[j].state["endogenous"]["Labor"] += 0

                            build.append(
                                {
                                    "builder": random_order_agents[i].idx,
                                    "type": "red_house_trade",
                                    "loc": np.array(random_order_agents[i].loc),
                                    "income": float(random_order_agents[i].state["build_payment_red_house"]),
                                }
                            )

                            build.append(
                                {
                                    "builder": random_order_agents[j].idx,
                                    "type": "red_house_trade",
                                    "loc": np.array(random_order_agents[i].loc),
                                    "income": float(random_order_agents[j].state["build_payment_red_house"]),
                                }
                            )

            # Sell/Buy a blue house if it is possible!
            elif random_order_agents_actions[i] == 5 and random_order_agents_expertise[i] == "Expert":
                agents_counted_action_trade_blue_house.append(random_order_agents[i])

                # Apply any building actions taken by the mobile agents
                for j in range(len(random_order_agents_actions)):

                    if random_order_agents_actions[j] == 5 and random_order_agents_expertise[j] == "Novice" and random_order_agents[j] not in agents_counted_action_trade_blue_house:

                        agents_counted_action_trade_blue_house.append(random_order_agents[j])

                        # Build and sell/buy a blue house if you can!
                        self.resource_cost = {"Wood": 1, "Iron": 1}
                
                        if self.agent_can_build(random_order_agents[i]):
                            # Remove the resources
                            for resource, cost in self.resource_cost.items():
                                random_order_agents[i].state["inventory"][resource] -= cost

                            # Place a house where the agent is standing
                            loc_r, loc_c = random_order_agents[i].loc
                            self.world.create_landmark("BlueHouse", loc_r, loc_c, random_order_agents[i].idx)

                            # Receive payment for the house
                            random_order_agents[i].state["inventory"]["Coin"] += random_order_agents[i].state["build_payment_blue_house"]
                            random_order_agents[j].state["inventory"]["Coin"] += random_order_agents[j].state["build_payment_blue_house"]

                            # Incur the labor cost for building
                            random_order_agents[i].state["endogenous"]["Labor"] += 0
                            random_order_agents[j].state["endogenous"]["Labor"] += 0

                            build.append(
                                {
                                    "builder": random_order_agents[i].idx,
                                    "type": "blue_house_trade",
                                    "loc": np.array(random_order_agents[i].loc),
                                    "income": float(random_order_agents[i].state["build_payment_blue_house"]),
                                }
                            )

                            build.append(
                                {
                                    "builder": random_order_agents[j].idx,
                                    "type": "blue_house_trade",
                                    "loc": np.array(random_order_agents[i].loc),
                                    "income": float(random_order_agents[j].state["build_payment_blue_house"]),
                                }
                            )

            # Sell/Buy a green house if it is possible!
            elif random_order_agents_actions[i] == 6 and random_order_agents_expertise[i] == "Expert":
                agents_counted_action_trade_green_house.append(random_order_agents[i])

                # Apply any building actions taken by the mobile agents
                for j in range(len(random_order_agents_actions)):

                    if random_order_agents_actions[j] == 6 and random_order_agents_expertise[j] == "Novice" and random_order_agents[j] not in agents_counted_action_trade_green_house:

                        agents_counted_action_trade_green_house.append(random_order_agents[j])

                        # Build and sell/buy a green house if you can!
                        self.resource_cost = {"Stone": 1, "Iron": 1}
                
                        if self.agent_can_build(random_order_agents[i]):
                            # Remove the resources
                            for resource, cost in self.resource_cost.items():
                                random_order_agents[i].state["inventory"][resource] -= cost

                            # Place a house where the agent is standing
                            loc_r, loc_c = random_order_agents[i].loc
                            self.world.create_landmark("GreenHouse", loc_r, loc_c, random_order_agents[i].idx)

                            # Receive payment for the house
                            random_order_agents[i].state["inventory"]["Coin"] += random_order_agents[i].state["build_payment_green_house"]
                            random_order_agents[j].state["inventory"]["Coin"] += random_order_agents[j].state["build_payment_green_house"]

                            # Incur the labor cost for building
                            random_order_agents[i].state["endogenous"]["Labor"] += 0
                            random_order_agents[j].state["endogenous"]["Labor"] += 0

                            build.append(
                                {
                                    "builder": random_order_agents[i].idx,
                                    "type": "green_house_trade",
                                    "loc": np.array(random_order_agents[i].loc),
                                    "income": float(random_order_agents[i].state["build_payment_green_house"]),
                                }
                            )

                            build.append(
                                {
                                    "builder": random_order_agents[j].idx,
                                    "type": "green_house_trade",
                                    "loc": np.array(random_order_agents[i].loc),
                                    "income": float(random_order_agents[j].state["build_payment_green_house"]),
                                }
                            )

            # Sell/Buy house building skill if it is possible!
            elif random_order_agents_actions[i] == 7 and random_order_agents_expertise[i] == "Expert":
                agents_counted_action_trade_build_skill.append(random_order_agents[i])

                # Apply any building actions taken by the mobile agents
                for j in range(len(random_order_agents_actions)):

                    if random_order_agents_actions[j] == 7 and random_order_agents_expertise[j] == "Novice" and random_order_agents[j] not in agents_counted_action_trade_build_skill:

                        agents_counted_action_trade_build_skill.append(random_order_agents[j])

                        # Receive payment for the house
                        random_order_agents[i].state["inventory"]["Coin"] += 0.5 * np.mean([random_order_agents[i].state["build_payment_red_house"], random_order_agents[i].state["build_payment_blue_house"], random_order_agents[i].state["build_payment_green_house"]])
                        random_order_agents[j].state["inventory"]["Coin"] += 0.5 * np.mean([random_order_agents[j].state["build_payment_red_house"], random_order_agents[j].state["build_payment_blue_house"], random_order_agents[j].state["build_payment_green_house"]])

                        random_order_agents[i].state["payment_max_house_building_skill_multiplier"] = random_order_agents[i].state["payment_max_house_building_skill_multiplier"] + 2.5
                        random_order_agents[j].state["payment_max_house_building_skill_multiplier"] = random_order_agents[j].state["payment_max_house_building_skill_multiplier"] + 2.5

                        # Incur the labor cost for building
                        random_order_agents[i].state["endogenous"]["Labor"] += 0
                        random_order_agents[j].state["endogenous"]["Labor"] += 0

                        build.append(
                            {
                                "builder": random_order_agents[i].idx,
                                "type": "build_skill_trade",
                                "loc": [0, 0],
                                "income": 0.5 * np.mean([random_order_agents[i].state["build_payment_red_house"], random_order_agents[i].state["build_payment_blue_house"], random_order_agents[i].state["build_payment_green_house"]]),
                            }
                        )

                        build.append(
                            {
                                "builder": random_order_agents[j].idx,
                                "type": "build_skill_trade",
                                "loc": [0, 0],
                                "income": 0.5 * np.mean([random_order_agents[j].state["build_payment_red_house"], random_order_agents[j].state["build_payment_blue_house"], random_order_agents[j].state["build_payment_green_house"]]),
                            }
                        )
                                           
        self.builds.append(build)

    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Here, agents observe their build skill. The planner does not observe anything
        from this component.
        """

        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[agent.idx] = {
                "build_payment_red_house": agent.state["build_payment_red_house"] / self.payment,
                "build_payment_blue_house": agent.state["build_payment_blue_house"] / self.payment,
                "build_payment_green_house": agent.state["build_payment_green_house"] / self.payment,
                "build_skill_red_house": agent.state["build_skill_red_house"],
                "build_skill_blue_house": agent.state["build_skill_blue_house"],
                "build_skill_green_house": agent.state["build_skill_green_house"],
                "payment_max_house_building_skill_multiplier": agent.state["payment_max_house_building_skill_multiplier"],
            }

        return obs_dict

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.

        Prevent building only if a landmark already occupies the agent's location.
        """

        masks = {}
        # Mobile agents' build action is masked if they cannot build with their current location and/or endowment
        for agent in self.world.agents:
            masks[agent.idx] = np.array([int(self.agent_can_build(agent)), int(self.agent_can_build(agent)), int(self.agent_can_build(agent)),
                                         int(self.agent_can_build(agent)), int(self.agent_can_build(agent)), int(self.agent_can_build(agent)),
                                         int(self.agent_can_build(agent))])

        return masks

    # For non-required customization
    # ------------------------------

    def get_metrics(self):
        """
        Metrics that capture what happened through this component.

        Returns:
            metrics (dict): A dictionary of {"metric_name": metric_value},
                where metric_value is a scalar.
        """
        
        world = self.world

        build_stats = {a.idx: {"n_builds": 0} for a in world.agents}
        for builds in self.builds:
            for build in builds:
                idx = build["builder"]
                build_stats[idx]["n_builds"] += 1

        out_dict = {}
        for a in world.agents:
            for k, v in build_stats[a.idx].items():
                out_dict["{}/{}".format(a.idx, k)] = v

        num_houses = np.sum(world.maps.get("RedHouse") > 0) + np.sum(world.maps.get("BlueHouse") > 0) + np.sum(world.maps.get("GreenHouse") > 0)
        out_dict["total_builds"] = num_houses

        return out_dict

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Re-sample agents' building skills.
        """

        self.sampled_skills_red_house = {agent.idx: 1 for agent in self.world.agents}
        self.sampled_skills_blue_house = {agent.idx: 1 for agent in self.world.agents}
        self.sampled_skills_green_house = {agent.idx: 1 for agent in self.world.agents}

        for agent in self.world.agents:
            PMSM = agent.state["payment_max_house_building_skill_multiplier"] 

            if self.skill_dist == "none":
                sampled_skill_red_house = 1
                sampled_skill_blue_house = 1
                sampled_skill_green_house = 1
                
                pay_rate_red = 1
                pay_rate_blue = 1
                pay_rate_green = 1
            elif self.skill_dist == "pareto":
                sampled_skill_red_house = np.random.pareto(4)
                sampled_skill_blue_house = np.random.pareto(4)
                sampled_skill_green_house = np.random.pareto(4)
                
                pay_rate_red = np.minimum(PMSM, (PMSM - 1) * sampled_skill_red_house + 1)
                pay_rate_blue = np.minimum(PMSM, (PMSM - 1) * sampled_skill_blue_house + 1)
                pay_rate_green = np.minimum(PMSM, (PMSM - 1) * sampled_skill_green_house + 1)
            elif self.skill_dist == "lognormal":
                sampled_skill_red_house = np.random.lognormal(-1, 0.5)
                sampled_skill_blue_house = np.random.lognormal(-1, 0.5)
                sampled_skill_green_house = np.random.lognormal(-1, 0.5)
                
                pay_rate_red = np.minimum(PMSM, (PMSM - 1) * sampled_skill_red_house + 1)
                pay_rate_blue = np.minimum(PMSM, (PMSM - 1) * sampled_skill_blue_house + 1)
                pay_rate_green = np.minimum(PMSM, (PMSM - 1) * sampled_skill_green_house + 1)
            else:
                raise NotImplementedError

            agent.state["build_payment_red_house"] = float(pay_rate_red * self.payment)
            agent.state["build_skill_red_house"] = float(sampled_skill_red_house)
            
            agent.state["build_payment_blue_house"] = float(pay_rate_blue * self.payment)
            agent.state["build_skill_blue_house"] = float(sampled_skill_blue_house)
            
            agent.state["build_payment_green_house"] = float(pay_rate_green * self.payment)
            agent.state["build_skill_green_house"] = float(sampled_skill_green_house)

            self.sampled_skills_red_house[agent.idx] = sampled_skill_red_house
            self.sampled_skills_blue_house[agent.idx] = sampled_skill_blue_house
            self.sampled_skills_green_house[agent.idx] = sampled_skill_green_house

        self.builds = []

    def get_dense_log(self):
        """
        Log builds.

        Returns:
            builds (list): A list of build events. Each entry corresponds to a single
                timestep and contains a description of any builds that occurred on
                that timestep.
        """
        
        return self.builds