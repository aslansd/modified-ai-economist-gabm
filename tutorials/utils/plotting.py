# Modified by Aslan Satary Dizaji, Copyright (c) 2024.

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from modified_ai_economist_gabm.foundation.entities.landmarks import RedHouse, BlueHouse, GreenHouse
from modified_ai_economist_gabm.foundation.entities.resources import Wood, Stone, Iron


def plot_map(maps, locs, ax=None, cmap_order=None):
    world_size = np.array(maps.get("Wood")).shape
    max_health = {"Wood": 1, "Stone": 1, "Iron": 1}
    n_agents = len(locs)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        ax.cla()
    
    tmp = np.zeros((3, world_size[0], world_size[1]))
    cmap = plt.get_cmap("jet", n_agents)

    if cmap_order is None:
        cmap_order = list(range(n_agents))
    else:
        cmap_order = list(cmap_order)
        assert len(cmap_order) == n_agents

    scenario_entities = [k for k in maps.keys() if "source" not in k.lower()]
    
    for entity in scenario_entities:
        if entity == "RedHouse":            
            try:
                map_ = (
                    RedHouse.color[0]
                    * np.array(maps.get("RedHouse"))
                )
                tmp[0, :, :] += map_  
            
            except Exception:
                map_ = (
                    RedHouse.color[0]
                    * np.array(maps.get("RedHouse")['health'])
                )
                tmp[0, :, :] += map_
        
        elif entity == "BlueHouse":
            try:
                map_ = (
                    BlueHouse.color[2]
                    * np.array(maps.get("BlueHouse"))
                )
                tmp[1, :, :] += map_
            
            except Exception:
                map_ = (
                    BlueHouse.color[2]
                    * np.array(maps.get("BlueHouse")['health'])
                )
                tmp[1, :, :] += map_

        elif entity == "GreenHouse":
            try:
                map_ = (
                    GreenHouse.color[1]
                    * np.array(maps.get("GreenHouse"))
                )
                tmp[2, :, :] += map_
            
            except Exception:
                map_ = (
                    BlueHouse.color[1]
                    * np.array(maps.get("GreenHouse")['health'])
                )
                tmp[2, :, :] += map_
                     
        elif entity == "Wood":
            if Wood.collectible:
                map_ = (
                    Wood.color[0]
                    * np.array(maps.get("Wood"))
                )
                map_ /= max_health["Wood"]
                tmp[0, :, :] += map_
        
        elif entity == "Stone":
            if Stone.collectible:
                map_ = (
                    Stone.color[1]
                    * np.array(maps.get("Stone"))
                )
                map_ /= max_health["Stone"]
                tmp[1, :, :] += map_
        
        elif entity == "Iron":
            if Iron.collectible:
                map_ = (
                    Iron.color[2]
                    * np.array(maps.get("Iron"))
                )
                map_ /= max_health["Iron"]
                tmp[2, :, :] += map_
        
        else:
            continue

    if isinstance(maps, dict):
        house_idx_red = np.array(maps.get("RedHouse")["owner"])
        house_health_red = np.array(maps.get("RedHouse")["health"])
        
        house_idx_blue = np.array(maps.get("BlueHouse")["owner"])
        house_health_blue = np.array(maps.get("BlueHouse")["health"])

        house_idx_green = np.array(maps.get("GreenHouse")["owner"])
        house_health_green = np.array(maps.get("GreenHouse")["health"])
    else:
        house_idx_red = maps.get("RedHouse", owner=True)
        house_health_red = maps.get("RedHouse")
        
        house_idx_blue = maps.get("BlueHouse", owner=True)
        house_health_blue = maps.get("BlueHouse")

        house_idx_green = maps.get("GreenHouse", owner=True)
        house_health_green = maps.get("GreenHouse")
    
    for i in range(n_agents):
        houses = house_health_red * (house_idx_red == cmap_order[i]) + house_health_blue * (house_idx_blue == cmap_order[i]) + house_health_green * (house_idx_green == cmap_order[i])
        agent = np.zeros_like(houses)
        agent += houses
        col = np.array(cmap(i)[:3])
        map_ = col[:, None, None] * agent[None]
        tmp += map_

    tmp *= 0.7
    tmp += 0.3

    tmp = np.transpose(tmp, [1, 2, 0])
    tmp = np.minimum(tmp, 1.0)

    ax.imshow(tmp, vmax=1.0, aspect="auto")

    bbox = ax.get_window_extent()

    for i in range(n_agents):
        r, c = locs[cmap_order[i]]
        col = np.array(cmap(i)[:3])
        ax.plot(c, r, "o", markersize=bbox.height * 20 / 550, color="w")
        ax.plot(c, r, "*", markersize=bbox.height * 15 / 550, color=col)

    ax.set_xticks([])
    ax.set_yticks([])


def plot_env_state(env, ax=None, remap_key=None):
    maps = env.world.maps
    locs = [agent.loc for agent in env.world.agents]

    if remap_key is None:
        cmap_order = None
    else:
        assert isinstance(remap_key, str)
        cmap_order = np.argsort(
            [agent.state[remap_key] for agent in env.world.agents]
        ).tolist()

    plot_map(maps, locs, ax, cmap_order)


def plot_log_state(dense_log, t, ax=None, remap_key=None):
    maps = dense_log["world"][t]
    states = dense_log["states"][t]

    n_agents = len(states) - 1
    locs = []
    for i in range(n_agents):
        r, c = states[str(i)]["loc"]
        locs.append([r, c])

    if remap_key is None:
        cmap_order = None
    else:
        assert isinstance(remap_key, str)
        key_val = np.array(
            [dense_log["states"][0][str(i)][remap_key] for i in range(n_agents)]
        )
        cmap_order = np.argsort(key_val).tolist()

    plot_map(maps, locs, ax, cmap_order)


def _format_logs_and_eps(dense_logs, eps):
    if isinstance(dense_logs, dict):
        return [dense_logs], [0]
    else:
        assert isinstance(dense_logs, (list, tuple))

    if isinstance(eps, (list, tuple)):
        return dense_logs, list(eps)
    elif isinstance(eps, (int, float)):
        return dense_logs, [int(eps)]
    elif eps is None:
        return dense_logs, list(range(np.minimum(len(dense_logs), 16)))
    else:
        raise NotImplementedError


def vis_world_array(dense_logs, ts, eps=None, axes=None, remap_key=None):
    dense_logs, eps = _format_logs_and_eps(dense_logs, eps)
    if isinstance(ts, (int, float)):
        ts = [ts]

    if axes is None:
        fig, axes = plt.subplots(
            len(eps),
            len(ts),
            figsize=(np.minimum(3.2 * len(ts), 16), 3 * len(eps)),
            squeeze=False,
        )

    else:
        fig = None

        if len(ts) == 1 and len(eps) == 1:
            axes = np.array([[axes]]).reshape(1, 1)
        else:
            try:
                axes = np.array(axes).reshape(len(eps), len(ts))
            except ValueError:
                print("Could not reshape provided axes array into the necessary shape!")
                raise

    for ti, t in enumerate(ts):
        for ei, ep in enumerate(eps):
            plot_log_state(dense_logs[ep], t, ax=axes[ei, ti], remap_key=remap_key)

    for ax, t in zip(axes[0], ts):
        ax.set_title("T = {}".format(t))
    for ax, ep in zip(axes[:, 0], eps):
        ax.set_ylabel("Episode {}".format(ep))

    return fig


def vis_world_range(dense_logs, t0=0, tN=None, N=5, eps=None, axes=None, remap_key=None):
    dense_logs, eps = _format_logs_and_eps(dense_logs, eps)

    viable_ts = np.array([i for i, w in enumerate(dense_logs[0]["world"]) if w])
    if tN is None:
        tN = viable_ts[-1]
    assert 0 <= t0 < tN
    target_ts = np.linspace(t0, tN, N).astype(np.int)

    ts = set()
    for tt in target_ts:
        closest = np.argmin(np.abs(tt - viable_ts))
        ts.add(viable_ts[closest])
    
    ts = sorted(list(ts))
    if axes is not None:
        axes = axes[: len(ts)]
    
    return vis_world_array(dense_logs, ts, axes=axes, eps=eps, remap_key=remap_key)


def vis_builds(dense_logs, eps=None, ax=None):
    dense_logs, eps = _format_logs_and_eps(dense_logs, eps)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 3))
    
    cmap = plt.get_cmap("jet", len(eps))
    
    for i, ep in enumerate(eps):
        ax.plot(
            np.cumsum([len(b["builds"]) for b in dense_logs[ep]["Build"]]),
            color=cmap(i),
            label="Ep {}".format(ep),
        )
    
    ax.legend()
    ax.grid(b=True)
    ax.set_ylim(bottom=0)


def trade_str(c_trades, resource, agent, income=True):
    if income:
        p = [x["income"] for x in c_trades[resource] if x["seller"] == agent]
    else:
        p = [x["cost"] for x in c_trades[resource] if x["buyer"] == agent]
    
    if len(p) > 0:
        return "{:6.2f} (n={:3d})".format(np.mean(p), len(p))
    else:
        tmp = "~" * 8
        tmp = (" ") * 3 + tmp + (" ") * 3
        return tmp


def full_trade_str(c_trades, resource, a_indices, income=True):
    s_head = "{} ({})".format("Income" if income else "Cost", resource)
    ac_strings = [trade_str(c_trades, resource, buyer, income) for buyer in a_indices]
    s_tail = " | ".join(ac_strings)
    
    return "{:<15}: {}".format(s_head, s_tail)


def build_str(all_builds, agent):
    p = [x["income"] for x in all_builds if x["builder"] == agent]
    
    if len(p) > 0:
        return "{:6.2f} (n={:3d})".format(np.mean(p), len(p))
    else:
        tmp = "~" * 8
        tmp = (" ") * 3 + tmp + (" ") * 3
        return tmp


def full_build_str(all_builds, a_indices):
    s_head = "Income (Build)"
    ac_strings = [build_str(all_builds, builder) for builder in a_indices]
    s_tail = " | ".join(ac_strings)
    
    return "{:<15}: {}".format(s_head, s_tail)


def header_str(n_agents):
    s_head = ("_" * 15) + ":_"
    s_tail = "_|_".join([" Agent {:2d} ____".format(i) for i in range(n_agents)])
    
    return s_head + s_tail


def report(c_trades, all_builds, n_agents, a_indices=None):
    if a_indices is None:
        a_indices = list(range(n_agents))
    
    print(header_str(n_agents))
    
    resources = ["Wood", "Stone", "Iron"]
    if c_trades is not None:
        for resource in resources:
            print(full_trade_str(c_trades, resource, a_indices, income=False))
        
        print("")
        
        for resource in resources:
            print(full_trade_str(c_trades, resource, a_indices, income=True))
    
    print(full_build_str(all_builds, a_indices))


def breakdown(log, remap_key=None, governance_type=None, episode_length=2000):
    fig0 = vis_world_range(log, remap_key=remap_key)

    n = len(list(log["states"][0].keys())) - 1
    trading_active = "Trade" in log

    if remap_key is None:
        aidx = list(range(n))
    else:
        assert isinstance(remap_key, str)
        key_vals = np.array([log["states"][0][str(i)][remap_key] for i in range(n)])
        aidx = np.argsort(key_vals).tolist()

    all_builds = []
    for t, builds in enumerate(log["BuildTradeHouseSkill"]):
        if isinstance(builds, dict):
            builds_ = builds["builds"]
        else:
            builds_ = builds
        for build in builds_:
            this_build = {"t": t}
            this_build.update(build)
            all_builds.append(this_build)
    
    if trading_active:
        c_trades = {"Wood": [], "Stone": [], "Iron": [], "Coin": []}
        for t, trades in enumerate(log["Trade"]):
            if isinstance(trades, dict):
                trades_ = trades["trades"]
            else:
                trades_ = trades
            for trade in trades_:
                this_trade = {
                    "t": t,
                    "t_ask": t - trade["ask_lifetime"],
                    "t_bid": t - trade["bid_lifetime"],
                }
                
                this_trade.update(trade)
                
                c_trades[trade["commodity"]].append(this_trade)   

            incomes = {
            "Sell Wood": [
                sum([t["income"] for t in c_trades["Wood"] if t["seller"] == aidx[i]])
                for i in range(n)
            ],
            "Buy Wood": [
                sum([-t["price"] for t in c_trades["Wood"] if t["buyer"] == aidx[i]])
                for i in range(n)
            ],
            "Sell Stone": [
                sum([t["income"] for t in c_trades["Stone"] if t["seller"] == aidx[i]])
                for i in range(n)
            ],
            "Buy Stone": [
                sum([-t["price"] for t in c_trades["Stone"] if t["buyer"] == aidx[i]])
                for i in range(n)
            ],
            "Sell Iron": [
                sum([t["income"] for t in c_trades["Iron"] if t["seller"] == aidx[i]])
                for i in range(n)
            ],
            "Buy Iron": [
                sum([-t["price"] for t in c_trades["Iron"] if t["buyer"] == aidx[i]])
                for i in range(n)
            ],
            "Build": [
                sum([b["income"] for b in all_builds if b["builder"] == aidx[i]])
                for i in range(n)
            ],
        }

    else:
        c_trades = None
        incomes = {
            "Build": [
                sum([b["income"] for b in all_builds if b["builder"] == aidx[i]])
                for i in range(n)
            ],
        }

    incomes["Total"] = np.stack([v for v in incomes.values()]).sum(axis=0)

    endows = [
        int(
            float(log["states"][-1][str(aidx[i])]["inventory"]["Coin"])
            + float(log["states"][-1][str(aidx[i])]["escrow"]["Coin"])
        )
        for i in range(n)
    ]

    n_small = np.minimum(6, n)

    report(c_trades, all_builds, n, aidx)

    cmap = plt.get_cmap("jet", n)
    rs = ["Wood", "Stone", "Iron", "Coin"]

    fig1, axes = plt.subplots(1, len(rs) + 1, figsize=(16, 4), sharey=False)
    for r, ax in zip(rs, axes):
        for i in range(n):
            ax.plot(
                [
                    log["states"][j][str(aidx[i])]["inventory"][r] + log["states"][j][str(aidx[i])]["escrow"][r]
                    for j in range(len(log["states"]))
                ],
                label=i,
                color=cmap(i),
            )           
        ax.set_title(r)
        ax.legend()
        ax.grid(b=True)

    ax = axes[-1]
    for i in range(n):
        ax.plot(
            [log["states"][j][str(aidx[i])]["endogenous"]["Labor"] for j in range(len(log["states"]))],
            label=i,
            color=cmap(i),
        )
    ax.set_title("Labor")
    ax.legend()
    ax.grid(b=True)

    tmp = np.array(log["world"][0]["Wood"])
    
    fig2, axes = plt.subplots(
        1,
        n_small,
        figsize=(16, 4),
        sharex="row",
        sharey="row",
        squeeze=False,
    )
    
    for i, ax in enumerate(axes[0]):
        rows = np.array([x[str(aidx[i])]["loc"][0] for x in log["states"]]) * -1
        cols = np.array([x[str(aidx[i])]["loc"][1] for x in log["states"]])
        ax.plot(cols[::20], rows[::20])
        ax.plot(cols[0], rows[0], "r*", markersize=15)
        ax.plot(cols[-1], rows[-1], "g*", markersize=15)
        ax.set_title("Agent {}".format(i))
        ax.set_xlim([-1, 1 + tmp.shape[1]])
        ax.set_ylim([-(1 + tmp.shape[0]), 1])

    if trading_active:
        
        fig3, axes = plt.subplots(
            1,
            n_small,
            figsize=(16, 4),
            sharex="row",
            sharey="row",
            squeeze=False,
        )
        
        for i, ax in enumerate(axes[0]):
            tmp = [
                (s["t"], s["income"]) for s in c_trades["Wood"] if s["seller"] == aidx[i]
            ]
            
            if tmp:
                ts, prices = [np.array(x) for x in zip(*tmp)]
                ax.plot(
                    np.stack([ts, ts]),
                    np.stack([np.zeros_like(prices), prices]),
                    color=Wood.color,
                )
                ax.plot(
                    ts, prices, ".", color=Wood.color, markersize=12
                )

            tmp = [
                (s["t"], -s["cost"]) for s in c_trades["Wood"] if s["buyer"] == aidx[i]
            ]
            
            if tmp:
                ts, prices = [np.array(x) for x in zip(*tmp)]
                ax.plot(
                    np.stack([ts, ts]),
                    np.stack([np.zeros_like(prices), prices]),
                    color=Wood.color,
                )
                ax.plot(
                    ts, prices, ".", color=Wood.color, markersize=12, label='Wood'
                )
            
            tmp = [
                (s["t"], s["income"]) for s in c_trades["Stone"] if s["seller"] == aidx[i]
            ]
            
            if tmp:
                ts, prices = [np.array(x) for x in zip(*tmp)]
                ax.plot(
                    np.stack([ts, ts]),
                    np.stack([np.zeros_like(prices), prices]),
                    color=Stone.color,
                )
                ax.plot(
                    ts, prices, ".", color=Stone.color, markersize=12
                )

            tmp = [
                (s["t"], -s["cost"]) for s in c_trades["Stone"] if s["buyer"] == aidx[i]
            ]
            
            if tmp:
                ts, prices = [np.array(x) for x in zip(*tmp)]
                ax.plot(
                    np.stack([ts, ts]),
                    np.stack([np.zeros_like(prices), prices]),
                    color=Stone.color,
                )
                ax.plot(
                    ts, prices, ".", color=Stone.color, markersize=12, label='Stone'
                )
            
            tmp = [
                (s["t"], s["income"]) for s in c_trades["Iron"] if s["seller"] == aidx[i]
            ]
            
            if tmp:
                ts, prices = [np.array(x) for x in zip(*tmp)]
                ax.plot(
                    np.stack([ts, ts]),
                    np.stack([np.zeros_like(prices), prices]),
                    color=Iron.color,
                )
                ax.plot(
                    ts, prices, ".", color=Iron.color, markersize=12,
                )

            tmp = [
                (s["t"], -s["cost"]) for s in c_trades["Iron"] if s["buyer"] == aidx[i]
            ]
            
            if tmp:
                ts, prices = [np.array(x) for x in zip(*tmp)]
                ax.plot(
                    np.stack([ts, ts]),
                    np.stack([np.zeros_like(prices), prices]),
                    color=Iron.color,
                )
                ax.plot(
                    ts, prices, ".", color=Iron.color, markersize=12, label='Iron'
                )
            
            ax.plot([-20, len(log["states"]) + 19], [0, 0], "w-")
            # ax.set_ylim([-10.2, 10.2])
            ax.set_xlim([-20, len(log["states"]) + 19])
            ax.grid(b=True)
            ax.set_facecolor([0.3, 0.3, 0.3])
            ax.set_title("Trades of Agent {}".format(i))
            ax.legend()
            
    all_votesinvests = []
    for t, votesinvests in enumerate(log["VoteInvest"][1]):
        if isinstance(builds, dict):
            votesinvests_ = votesinvests["votesinvests"]
        else:
            votesinvests_ = votesinvests
        for voteinvest in votesinvests_:
            this_voteinvest = {"t": t}
            this_voteinvest.update(voteinvest)
            all_votesinvests.append(this_voteinvest)
    
    if governance_type == "AgentAgent":
        fig4, axes = plt.subplots(
            1,
            n_small,
            figsize=(16, 4),
            sharex="row",
            sharey="row",
            squeeze=False,
        )
        
        for i, ax in enumerate(axes[0]):
            tmp = [
                (s["t"], s["vote"]) for s in all_votesinvests if s["voterinvestor"] == aidx[i]
            ]
            
            if tmp:
                ts = np.zeros(len(tmp), int)              
                
                borda_score = np.zeros(len(tmp), int)
                
                for j in range(len(tmp)):
                    ts[j] = tmp[j][0]
                    
                    if tmp[j][1][0] == "Wood":
                        borda_score[j] = 2
                    if tmp[j][1][1] == "Wood":
                        borda_score[j] = 1 
                    if tmp[j][1][2] == "Wood":
                        borda_score[j] = 0 
                
                ax.plot(
                    np.stack([ts, ts]),
                    np.stack([np.zeros_like(borda_score), borda_score]),
                    color=Wood.color,
                )
                ax.plot(
                    ts, borda_score, ".", color=Wood.color, markersize=12, label='Wood'
                )

                borda_score = np.zeros(len(tmp), int)

                for j in range(len(tmp)):
                    if tmp[j][1][0] == "Stone":
                        borda_score[j] = 2
                    if tmp[j][1][1] == "Stone":
                        borda_score[j] = 1
                    if tmp[j][1][2] == "Stone":
                        borda_score[j] = 0
                
                ax.plot(
                    np.stack([ts, ts]),
                    np.stack([np.zeros_like(borda_score), borda_score]),
                    color=Stone.color,
                )
                ax.plot(
                    ts, borda_score, ".", color=Stone.color, markersize=12, label='Stone'
                )

                borda_score = np.zeros(len(tmp), int)
                
                for j in range(len(tmp)):
                    if tmp[j][1][0] == "Iron":
                        borda_score[j] = 2
                    if tmp[j][1][1] == "Iron":
                        borda_score[j] = 1 
                    if tmp[j][1][2] == "Iron":
                        borda_score[j] = 0
                
                ax.plot(
                    np.stack([ts, ts]),
                    np.stack([np.zeros_like(borda_score), borda_score]),
                    color=Iron.color,
                )
                ax.plot(
                    ts, borda_score, ".", color=Iron.color, markersize=12, label='Iron'
                )
                    
            ax.plot([-1, ts[-1]], [0, 0], "w-")
            ax.set_ylim([0, 3])
            ax.set_xlim([-1, ts[-1]])
            ax.grid(b=True)
            ax.set_facecolor([0.3, 0.3, 0.3])
            ax.set_title("Votes of Agent {}".format(i))
            ax.legend()
            
        variables_matrix = np.zeros((episode_length, n, 13))
        
        for i in range(n):           
            tmp_1 = [
                (s["t"], s["income"]) for s in c_trades["Wood"] if (s["seller"] == aidx[i] or s["buyer"] == aidx[i])
            ]
            
            tmp_2 = [
                (s["t"], s["income"]) for s in c_trades["Stone"] if (s["seller"] == aidx[i] or s["buyer"] == aidx[i])
            ]
            
            tmp_3 = [
                (s["t"], s["income"]) for s in c_trades["Iron"] if (s["seller"] == aidx[i] or s["buyer"] == aidx[i])
            ]
            
            tmp_4 = [
                (s["t"], s["type"], s["income"]) for s in all_builds if s["builder"] == aidx[i]
            ]
            
            tmp_5 = [
                (s["t"], s["vote"]) for s in all_votesinvests if s["voterinvestor"] == aidx[i]
            ]
            
            for j in range(len(tmp_1)):
                variables_matrix[tmp_1[j][0]][i][0] = tmp_1[j][1]
                    
            for j in range(len(tmp_2)):
                variables_matrix[tmp_2[j][0]][i][1] = tmp_2[j][1]
            
            for j in range(len(tmp_3)):
                variables_matrix[tmp_3[j][0]][i][2] = tmp_3[j][1]
                                      
            for j in range(len(tmp_4)):
                if tmp_4[j][1] == 'red_house':
                    variables_matrix[tmp_4[j][0]][i][3] = tmp_4[j][2]
                elif tmp_4[j][1] == 'blue_house':
                    variables_matrix[tmp_4[j][0]][i][4] = tmp_4[j][2]
                elif tmp_4[j][1] == 'green_house':
                    variables_matrix[tmp_4[j][0]][i][5] = tmp_4[j][2]
                elif tmp_4[j][1] == 'red_house_trade':
                    variables_matrix[tmp_4[j][0]][i][6] = tmp_4[j][2]
                elif tmp_4[j][1] == 'blue_house_trade':
                    variables_matrix[tmp_4[j][0]][i][7] = tmp_4[j][2]
                elif tmp_4[j][1] == 'green_house_trade':
                    variables_matrix[tmp_4[j][0]][i][8] = tmp_4[j][2]
                elif tmp_4[j][1] == 'build_skill_trade':
                    variables_matrix[tmp_4[j][0]][i][9] = tmp_4[j][2]
                
            for j in range(len(tmp_5)):
                if tmp_5[j][1] == ["Wood", "Stone", "Iron"]:
                    variables_matrix[tmp_5[j][0]][i][10:] = [3, 2, 1]
                elif tmp_5[j][1] == ["Wood", "Iron", "Stone"]:
                    variables_matrix[tmp_5[j][0]][i][10:] = [3, 1, 2]
                elif tmp_5[j][1] == ["Stone", "Wood", "Iron"]:
                    variables_matrix[tmp_5[j][0]][i][10:] = [2, 3, 1]
                elif tmp_5[j][1] == ["Stone", "Iron", "Wood"]:
                    variables_matrix[tmp_5[j][0]][i][10:] = [2, 1, 3]
                elif tmp_5[j][1] == ["Iron", "Wood", "Stone"]:
                    variables_matrix[tmp_5[j][0]][i][10:] = [1, 3, 2]
                elif tmp_5[j][1] == ["Iron", "Stone", "Wood"]:
                    variables_matrix[tmp_5[j][0]][i][10:] = [1, 2, 3]
            
        fig0.savefig('AgentAgent/Visualizaion of the World_AgentAgent.png')
        fig1.savefig('AgentAgent/Budget of Resources_AgentAgent.png')
        fig2.savefig('AgentAgent/Movement of Agents_AgentAgent.png')
        fig3.savefig('AgentAgent/Trading of Agents_AgentAgent.png')
        fig4.savefig('AgentAgent/Counted Votes_AgentAgent.png')
            
        return (fig0, fig1, fig2, fig3, fig4), incomes, endows, c_trades, all_builds, all_votesinvests, variables_matrix
    
    elif governance_type == "AgentPlanner":           
        fig4, axes = plt.subplots(
            1,
            1,
            figsize=(32, 8),
            sharex="row",
            sharey="row",
            squeeze=False,
        )
        
        for i, ax in enumerate(axes[0]):
            tmp = [
                (s["t"], s["vote"]) for s in all_votesinvests
            ]
            
            if tmp:
                ts = np.zeros(len(tmp), int)              
                
                borda_score = np.zeros(len(tmp), int)
                
                for j in range(len(tmp)):
                    ts[j] = tmp[j][0]
                    
                    if tmp[j][1][0] == "Wood":
                        borda_score[j] = 2
                    if tmp[j][1][1] == "Wood":
                        borda_score[j] = 1 
                    if tmp[j][1][2] == "Wood":
                        borda_score[j] = 0 
                
                ax.plot(
                    np.stack([ts, ts]),
                    np.stack([np.zeros_like(borda_score), borda_score]),
                    color=Wood.color,
                )
                ax.plot(
                    ts, borda_score, ".", color=Wood.color, markersize=12, label='Wood'
                )

                borda_score = np.zeros(len(tmp), int)

                for j in range(len(tmp)):
                    if tmp[j][1][0] == "Stone":
                        borda_score[j] = 2
                    if tmp[j][1][1] == "Stone":
                        borda_score[j] = 1
                    if tmp[j][1][2] == "Stone":
                        borda_score[j] = 0
                
                ax.plot(
                    np.stack([ts, ts]),
                    np.stack([np.zeros_like(borda_score), borda_score]),
                    color=Stone.color,
                )
                ax.plot(
                    ts, borda_score, ".", color=Stone.color, markersize=12, label='Stone'
                )

                borda_score = np.zeros(len(tmp), int)
                
                for j in range(len(tmp)):
                    if tmp[j][1][0] == "Iron":
                        borda_score[j] = 2
                    if tmp[j][1][1] == "Iron":
                        borda_score[j] = 1 
                    if tmp[j][1][2] == "Iron":
                        borda_score[j] = 0
                
                ax.plot(
                    np.stack([ts, ts]),
                    np.stack([np.zeros_like(borda_score), borda_score]),
                    color=Iron.color,
                )
                ax.plot(
                    ts, borda_score, ".", color=Iron.color, markersize=12, label='Iron'
                )
                    
            ax.plot([-1, ts[-1]], [0, 0], "w-")
            ax.set_ylim([0, 3])
            ax.set_xlim([-1, ts[-1]])
            ax.grid(b=True)
            ax.set_facecolor([0.3, 0.3, 0.3])
            ax.set_title("Planner's Counted Votes", fontsize=20)
            ax.legend()
            
        variables_matrix = np.zeros((episode_length, n, 13))
        
        for i in range(n):           
            tmp_1 = [
                (s["t"], s["income"]) for s in c_trades["Wood"] if (s["seller"] == aidx[i] or s["buyer"] == aidx[i])
            ]
            
            tmp_2 = [
                (s["t"], s["income"]) for s in c_trades["Stone"] if (s["seller"] == aidx[i] or s["buyer"] == aidx[i])
            ]
            
            tmp_3 = [
                (s["t"], s["income"]) for s in c_trades["Iron"] if (s["seller"] == aidx[i] or s["buyer"] == aidx[i])
            ]
            
            tmp_4 = [
                (s["t"], s["type"], s["income"]) for s in all_builds if s["builder"] == aidx[i]
            ]
            
            tmp_5 = [
                (s["t"], s["vote"]) for s in all_votesinvests
            ]
            
            for j in range(len(tmp_1)):
                variables_matrix[tmp_1[j][0]][i][0] = tmp_1[j][1]
                    
            for j in range(len(tmp_2)):
                variables_matrix[tmp_2[j][0]][i][1] = tmp_2[j][1]
            
            for j in range(len(tmp_3)):
                variables_matrix[tmp_3[j][0]][i][2] = tmp_3[j][1]
                                      
            for j in range(len(tmp_4)):
                if tmp_4[j][1] == 'red_house':
                    variables_matrix[tmp_4[j][0]][i][3] = tmp_4[j][2]
                elif tmp_4[j][1] == 'blue_house':
                    variables_matrix[tmp_4[j][0]][i][4] = tmp_4[j][2]
                elif tmp_4[j][1] == 'green_house':
                    variables_matrix[tmp_4[j][0]][i][5] = tmp_4[j][2]
                elif tmp_4[j][1] == 'red_house_trade':
                    variables_matrix[tmp_4[j][0]][i][6] = tmp_4[j][2]
                elif tmp_4[j][1] == 'blue_house_trade':
                    variables_matrix[tmp_4[j][0]][i][7] = tmp_4[j][2]
                elif tmp_4[j][1] == 'green_house_trade':
                    variables_matrix[tmp_4[j][0]][i][8] = tmp_4[j][2]
                elif tmp_4[j][1] == 'build_skill_trade':
                    variables_matrix[tmp_4[j][0]][i][9] = tmp_4[j][2]
                
            for j in range(len(tmp_5)):
                if tmp_5[j][1] == ["Wood", "Stone", "Iron"]:
                    variables_matrix[tmp_5[j][0]][i][10:] = [3, 2, 1]
                elif tmp_5[j][1] == ["Wood", "Iron", "Stone"]:
                    variables_matrix[tmp_5[j][0]][i][10:] = [3, 1, 2]
                elif tmp_5[j][1] == ["Stone", "Wood", "Iron"]:
                    variables_matrix[tmp_5[j][0]][i][10:] = [2, 3, 1]
                elif tmp_5[j][1] == ["Stone", "Iron", "Wood"]:
                    variables_matrix[tmp_5[j][0]][i][10:] = [2, 1, 3]
                elif tmp_5[j][1] == ["Iron", "Wood", "Stone"]:
                    variables_matrix[tmp_5[j][0]][i][10:] = [1, 3, 2]
                elif tmp_5[j][1] == ["Iron", "Stone", "Wood"]:
                    variables_matrix[tmp_5[j][0]][i][10:] = [1, 2, 3]
                
        fig0.savefig('AgentPlanner/Visualizaion of the World_AgentPlanner.png')
        fig1.savefig('AgentPlanner/Budget of Resources_AgentPlanner.png')
        fig2.savefig('AgentPlanner/Movement of Agents_AgentPlanner.png')
        fig3.savefig('AgentPlanner/Trading of Agents_AgentPlanner.png')
        fig4.savefig('AgentPlanner/Counted Votes_AgentPlanner.png')
   
        return (fig0, fig1, fig2, fig3, fig4), incomes, endows, c_trades, all_builds, all_votesinvests, variables_matrix
    
    elif governance_type == "PlannerPlanner":           
        fig4, axes = plt.subplots(
            1,
            1,
            figsize=(32, 8),
            sharex="row",
            sharey="row",
            squeeze=False,
        )
        
        for i, ax in enumerate(axes[0]):
            tmp = [
                (s["t"], s["vote"]) for s in all_votesinvests
            ]
            
            if tmp:
                ts = np.zeros(len(tmp), int)              
                
                borda_score = np.zeros(len(tmp), int)
                
                for j in range(len(tmp)):
                    ts[j] = tmp[j][0]
                    
                    if tmp[j][1][0] == "Wood":
                        borda_score[j] = 2
                    if tmp[j][1][1] == "Wood":
                        borda_score[j] = 1 
                    if tmp[j][1][2] == "Wood":
                        borda_score[j] = 0 
                
                ax.plot(
                    np.stack([ts, ts]),
                    np.stack([np.zeros_like(borda_score), borda_score]),
                    color=Wood.color,
                )
                ax.plot(
                    ts, borda_score, ".", color=Wood.color, markersize=12, label='Wood'
                )

                borda_score = np.zeros(len(tmp), int)

                for j in range(len(tmp)):
                    if tmp[j][1][0] == "Stone":
                        borda_score[j] = 2
                    if tmp[j][1][1] == "Stone":
                        borda_score[j] = 1
                    if tmp[j][1][2] == "Stone":
                        borda_score[j] = 0
                
                ax.plot(
                    np.stack([ts, ts]),
                    np.stack([np.zeros_like(borda_score), borda_score]),
                    color=Stone.color,
                )
                ax.plot(
                    ts, borda_score, ".", color=Stone.color, markersize=12, label='Stone'
                )

                borda_score = np.zeros(len(tmp), int)
                
                for j in range(len(tmp)):
                    if tmp[j][1][0] == "Iron":
                        borda_score[j] = 2
                    if tmp[j][1][1] == "Iron":
                        borda_score[j] = 1 
                    if tmp[j][1][2] == "Iron":
                        borda_score[j] = 0
                
                ax.plot(
                    np.stack([ts, ts]),
                    np.stack([np.zeros_like(borda_score), borda_score]),
                    color=Iron.color,
                )
                ax.plot(
                    ts, borda_score, ".", color=Iron.color, markersize=12, label='Iron'
                )
                    
            ax.plot([-1, ts[-1]], [0, 0], "w-")
            ax.set_ylim([0, 3])
            ax.set_xlim([-1, ts[-1]])
            ax.grid(b=True)
            ax.set_facecolor([0.3, 0.3, 0.3])
            ax.set_title("Planner's Votes", fontsize=20)
            ax.legend()
            
        variables_matrix = np.zeros((episode_length, n, 13))
        
        for i in range(n):           
            tmp_1 = [
                (s["t"], s["income"]) for s in c_trades["Wood"] if (s["seller"] == aidx[i] or s["buyer"] == aidx[i])
            ]
            
            tmp_2 = [
                (s["t"], s["income"]) for s in c_trades["Stone"] if (s["seller"] == aidx[i] or s["buyer"] == aidx[i])
            ]
            
            tmp_3 = [
                (s["t"], s["income"]) for s in c_trades["Iron"] if (s["seller"] == aidx[i] or s["buyer"] == aidx[i])
            ]
            
            tmp_4 = [
                (s["t"], s["type"], s["income"]) for s in all_builds if s["builder"] == aidx[i]
            ]
            
            tmp_5 = [
                (s["t"], s["vote"]) for s in all_votesinvests
            ]
            
            for j in range(len(tmp_1)):
                variables_matrix[tmp_1[j][0]][i][0] = tmp_1[j][1]
                    
            for j in range(len(tmp_2)):
                variables_matrix[tmp_2[j][0]][i][1] = tmp_2[j][1]
            
            for j in range(len(tmp_3)):
                variables_matrix[tmp_3[j][0]][i][2] = tmp_3[j][1]
                                      
            for j in range(len(tmp_4)):
                if tmp_4[j][1] == 'red_house':
                    variables_matrix[tmp_4[j][0]][i][3] = tmp_4[j][2]
                elif tmp_4[j][1] == 'blue_house':
                    variables_matrix[tmp_4[j][0]][i][4] = tmp_4[j][2]
                elif tmp_4[j][1] == 'green_house':
                    variables_matrix[tmp_4[j][0]][i][5] = tmp_4[j][2]
                elif tmp_4[j][1] == 'red_house_trade':
                    variables_matrix[tmp_4[j][0]][i][6] = tmp_4[j][2]
                elif tmp_4[j][1] == 'blue_house_trade':
                    variables_matrix[tmp_4[j][0]][i][7] = tmp_4[j][2]
                elif tmp_4[j][1] == 'green_house_trade':
                    variables_matrix[tmp_4[j][0]][i][8] = tmp_4[j][2]
                elif tmp_4[j][1] == 'build_skill_trade':
                    variables_matrix[tmp_4[j][0]][i][9] = tmp_4[j][2]
                
            for j in range(len(tmp_5)):
                if tmp_5[j][1] == ["Wood", "Stone", "Iron"]:
                    variables_matrix[tmp_5[j][0]][i][10:] = [3, 2, 1]
                elif tmp_5[j][1] == ["Wood", "Iron", "Stone"]:
                    variables_matrix[tmp_5[j][0]][i][10:] = [3, 1, 2]
                elif tmp_5[j][1] == ["Stone", "Wood", "Iron"]:
                    variables_matrix[tmp_5[j][0]][i][10:] = [2, 3, 1]
                elif tmp_5[j][1] == ["Stone", "Iron", "Wood"]:
                    variables_matrix[tmp_5[j][0]][i][10:] = [2, 1, 3]
                elif tmp_5[j][1] == ["Iron", "Wood", "Stone"]:
                    variables_matrix[tmp_5[j][0]][i][10:] = [1, 3, 2]
                elif tmp_5[j][1] == ["Iron", "Stone", "Wood"]:
                    variables_matrix[tmp_5[j][0]][i][10:] = [1, 2, 3]
        
        fig0.savefig('PlannerPlanner/Visualizaion of the World_PlannerPlanner.png')
        fig1.savefig('PlannerPlanner/Budget of Resources_PlannerPlanner.png')
        fig2.savefig('PlannerPlanner/Movement of Agents_PlannerPlanner.png')
        fig3.savefig('PlannerPlanner/Trading of Agents_PlannerPlanner.png')
        fig4.savefig('PlannerPlanner/Counted Votes_PlannerPlanner.png')
            
        return (fig0, fig1, fig2, fig3, fig4), incomes, endows, c_trades, all_builds, all_votesinvests, variables_matrix

def plot_for_each_n(y_fun, n, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    cmap = plt.get_cmap("jet", n)
    
    for i in range(n):
        ax.plot(y_fun(i), color=cmap(i), label=i)
    
    ax.legend()
    ax.grid(b=True)