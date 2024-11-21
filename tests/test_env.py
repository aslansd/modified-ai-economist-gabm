# Modified by Aslan Satary Dizaji, Copyright (c) 2024.

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
Unit tests for the wood and stone and iron and soil scenario + basic components
"""

import unittest
import numpy as np

from modified_ai_economist_gabm import foundation


class CreateEnv:
    """
    Create an environment instance based on a configuration
    """

    def __init__(self):
        self.env = None
        self.set_env_config()

    def set_env_config(self):
        """Set up a sample environment config"""
        self.env_config = {
            # ===== SCENARIO CLASS =====  
            # Which Scenario class to use: the class's name in the Scenario Registry (foundation.scenarios).
            # The environment object will be an instance of the Scenario class.
            'scenario_name' : 'uniform_scenario_for_vote_and_invest',
            
            # ===== COMPONENTS =====   
            # Which components to use (specified as list of ("component_name", {component_kwargs}) tuples).
            # "component_name" refers to the Component class's name in the Component Registry (foundation.components).
            # {component_kwargs} is a dictionary of kwargs passed to the Component class.
            # The order in which components reset, step, and generate obs follows their listed order below.
            'components' : [
                # (1) Trading collectible resources
                {'ContinuousDoubleAuction' : {
                    'max_bid_ask' :    10,
                    'order_labor' :    0.25,
                    'order_duration' : 50,
                    'max_num_orders' : 5,
                }},
                # (2) Movement and resource gathering
                {'Gather' : {
                    'move_labor' :     1,
                    'collect_labor' :  1,
                    'skill_dist' :     'none',
                }},
                # (3) Teaching and building houses
                {'TeachBuild' : {
                    'payment' :                      10,
                    'payment_max_skill_multiplier' : np.array([5, 5, 5]),
                    'skill_dist' :                   'pareto',
                    'build_labor' :                  10,
                }},
                # (4) Vote and invest resources
                {'AgentVotesAgentInvestsResources' : {
                    'vote_count_method' :        'Borda',
                    
                    # Similar to components/redistribution/PeriodicBracketTax
                    'disable_taxes' :            False,
                    'tax_model' :                'model_wrapper',
                    'period' :                   100,
                    'rate_min' :                 0.0,
                    'rate_max' :                 1.0,
                    'rate_disc' :                0.05,
                    'n_brackets' :               7,
                    'top_bracket_cutoff' :       100,
                    'usd_scaling' :              1000.0,
                    'bracket_spacing' :          'us-federal',
                    'fixed_bracket_rates' :      None,
                    'pareto_weight_type' :       'inverse_income',
                    'saez_fixed_elas' :          None,
                    'tax_annealing_schedule' :   None,
                }},
            ],
            
            # ===== STANDARD ARGUMENTS ======  
            # kwargs that are used by every Scenario class (i.e. defined in BaseEnvironment).
            'n_agents' :                          6,           # Number of non-planner agents (must be > 1)
            'world_size' :                        [42, 42],    # [Height, Width] of the env world
            'episode_length' :                    2000,        # Number of timesteps per episode
            
            # In multi-action-mode, the policy selects an action for each action subspace (defined in component code).
            # Otherwise, the policy selects only 1 action.
            'multi_action_mode_agents' :          False,                                                 
            'multi_action_mode_planner' :         True,    
            
            # When flattening observations, concatenate scalar & vector observations before output.
            # Otherwise, return observations with minimal processing.
            'flatten_observations' :              True,
            
            # When Flattening masks, concatenate each action subspace mask into a single array.
            # Note: flatten_masks = True is required for masking action logits in the code below.
            'flatten_masks' :                     True,
            
            'allow_observation_scaling' :         True,
            
            # How often to save the dense logs.
            'dense_log_frequency' :               20,
            
            'world_dense_log_frequency' :         50,
            'collate_agent_step_and_reset_data' : False,
            'seed' :                              None,
            
            # ===== SCENARIO CLASS ARGUMENTS =====   
            # Other arguments that are added by the Scenario class (i.e. not defined in BaseEnvironment).  
            'planner_gets_spatial_info' :         True,
            'full_observability' :                False,
            'mobile_agent_observation_range' :    5,
            'starting_wood_coverage' :            0.05,
            'wood_regen_halfwidth' :              0.5,
            'wood_regen_weight' :                 0.05,
            'wood_max_health' :                   1,
            'wood_clumpiness' :                   0.25,
            'starting_stone_coverage' :           0.05,
            'stone_regen_halfwidth' :             0.5,
            'stone_regen_weight' :                0.05,
            'stone_max_health' :                  1,
            'stone_clumpiness' :                  0.25,
            'starting_iron_coverage' :            0.05,
            'iron_regen_halfwidth' :              0.5,
            'iron_regen_weight' :                 0.05,
            'iron_max_health' :                   1,
            'iron_clumpiness' :                   0.25,
            'starting_soil_coverage' :            0.05,
            'soil_regen_halfwidth' :              0.5,
            'soil_regen_weight' :                 0.05,
            'soil_max_health' :                   1,
            'soil_clumpiness' :                   0.25,
            'gradient_steepness' :                8,
            'checker_source_blocks' :             False,
            'starting_agent_coin' :               10,
            'isoelastic_eta' :                    0.23,
            'energy_cost' :                       0.21,
            'energy_warmup_constant' :            0,
            'energy_warmup_method' :              'decay',
            'planner_reward_type' :               'coin_maximin_times_productivity',
            'mixing_weight_gini_vs_coin' :        0.0,
            'mixing_weight_maximin_vs_coin' :     0.0,
        }

        # Create an environment instance from the config
        self.env = foundation.make_env_instance(**self.env_config)


class TestEnv(unittest.TestCase):
    """Unit test to test the env wrapper, reset and step"""

    def test_env_reset_and_step(self):
        """
        Unit tests for the reset and step calls
        """
        
        create_env = CreateEnv()
        env = create_env.env

        # Assert that the total number of agents matches the sum of the 'n_agents' configuration and the number of planners (1 in this case).
        num_planners = 1
        self.assertEqual(
            len(env.all_agents), create_env.env_config["n_agents"] + num_planners
        )

        # Assert that the number of agents created in the world matches the configuration specification.
        self.assertEqual(len(env.world.agents), create_env.env_config["n_agents"])

        # Assert that the planner's index in the world is 'p'.
        self.assertEqual(env.world.planner.idx, "p")

        obs = env.reset()

        # Test whether the observation dictionary keys are created as expected
        self.assertEqual(
            sorted(list(obs.keys())),
            [str(i) for i in range(create_env.env_config["n_agents"])] + ["p"],
        )

        obs, reward, done, info = env.step({})

        # Check that the observation, reward and info keys match
        self.assertEqual(obs.keys(), reward.keys())
        self.assertEqual(obs.keys(), info.keys())

        # Assert that __all__ is in done
        assert "__all__" in done


if __name__ == "__main__":
    unittest.main()