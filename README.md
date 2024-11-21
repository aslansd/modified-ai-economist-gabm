## The extended AI-Economist which will be compared with the extended Concordia, a generative agent-based model (GABM)
## Incentives to Build Houses, Trade Houses, or Trade House Building Skills in Simulated Worlds under Various Governing Systems or Institutions: Comparing Multi-agent Reinforcement Learning to Generative Agent-based Model

- [The Paper](https://drive.google.com/file/d/1Tz5spfNeCmmC1ZFrkxa83CpKrrKKKEaT/view?usp=drive_link)
- [The Results](https://github.com/aslansd/modified-ai-economist-wc/tree/main/tutorials/Results)
- [The Notebook for the Full-Libertarian Governing System](https://github.com/aslansd/modified-ai-economist-wc/blob/main/tutorials/a_multi-agent_reinforcement_learning_study_of_evolution_of_communication_under_libertarian_and_utilitarian_governing_systems_RLlib_FullLibertarian.ipynb)
- [The Notebook for the Semi-Libertarian/Utilitarian Governing System](https://github.com/aslansd/modified-ai-economist-wc/blob/main/tutorials/a_multi-agent_reinforcement_learning_study_of_evolution_of_communication_under_libertarian_and_utilitarian_governing_systems_RLlib_SemiLibertarianUtilitarian.ipynb)
- [The Notebook for the Full-Utilitarian Governing System](https://github.com/aslansd/modified-ai-economist-wc/blob/main/tutorials/a_multi-agent_reinforcement_learning_study_of_evolution_of_communication_under_libertarian_and_utilitarian_governing_systems_RLlib_FullUtilitarian.ipynb)
- [The Notebook for Generating the Plots of the Paper](https://github.com/aslansd/modified-ai-economist-wc/blob/main/tutorials/a_multi-agent_reinforcement_learning_study_of_evolution_of_communication_under_libertarian_and_utilitarian_governing_systems_Plotting.ipynb)

### Abstract

It has been shown that social institutions impact human motivations to produce different behaviours. Governing system as one major social institution is able to incentivise people in a society to work more or less, specialise labor in one specific field, or diversify their types of earnings. Until recently, this type of investigation is normally performed via economists by building mathematical models or performing experiments in the field. However, with advancement in artificial intelligence (AI), now it is possible to perform in-silico simulations to test various hypotheses around this topic. Here, in a curiosity-driven project, I simulate two somewhat similar worlds using multi-agent reinforcement learning (MARL) framework of the AI-Economist and generative agent-based model (GABM) framework of the Concordia. The AI-Economist is a two-level MARL framework originally devised for simulating tax behaviours of agents in a society governed by a central planner. Here, I extend the AI-Economist so the agents beside being able to build houses using material resources of the environment, would be able to trade their built houses, or trade their house building skill. Moreover, I equip the agents and the central planner with a voting mechanism so they would be able to rank different material resources in the environment. As a result of these changes, I am able to generate two sets of governmental types. Along the individualistic-collectivists axis, I produce a set of three governing systems: Full-Libertarian, Semi-Libertarian/Utilitarian, and Full-Utilitarian. Additionally, I further divide the Semi-Libertarian/Utilitarian governing system along the discriminative axis to a set of three governing institutions: Inclusive, Arbitrary, and Extractive. Building on these, I am able to show that among three governing systems, under the Semi-Libertarian/Utilitarian one (closely resembling the current democratic governments in the world in which the agents vote and the government counts the votes of the agents and implements them accordingly), the ratios of building houses to trading houses and trading house building skill are higher than the rest. Similarly, among governing institutions of the Semi-Libertarian/Utilitarian governing system, under the Inclusive institution, the ratios of building houses to trading houses and trading house building skill are higher than the rest. Moreover, the GABM framework of Concordia is originally devised to facilitate construction and use of generative agent-based models to simulate interactions of agents in grounded social space. The agents of this framework perform actions using natural language of large language models (LLMs), and a special agent called game master (which its role is similar to the central planner in the AI-Economist) translates their actions into appropriate implementations. I extended this framework via a component considering the inventory, skill, build, vote, and tax of the agents simultaneously, to generate similar three governing systems as above: Full-Libertarian, Semi-Libertarian/Utilitarian, and Full-Utilitarian. Among these governing systems, when the game master cares about equality in the society, it seems that under the Full-Utilitarian one, the agents build more houses and trade more house building skill. In contrast, when the game master cares about productivity in the society, under the Full-Libertarian governing system, it seems that the agents simultaneously build more houses, trade more houses, and trade more house building skill. Overall, the focus of this paper is on exploratory modelling and comparison of the power of two advanced techniques of AI, MARL and GABM, to simulate a similar social phenomena with limitations. Thus its main findings need further evaluation to be strengthen.

<p align="center">
  <img width="800" src="figs/Slide_1.jpeg" />
</p>

## Installation Instructions

This package was installed and tested in Windows 10 Pro. To install this package from the source, you'll need to have Python 3.7 installed:

1. Clone this repository to your local machine:

 ```
   git clone www.github.com/aslansd/modified-ai-economist-gabm
 ```

2. Create a new conda environment (named "modified-ai-economist-gabm" below - replace with anything else) and activate it:

```pyfunctiontypecomment
   conda create --name modified-ai-economist-gabm python=3.7 --yes
   conda activate modified-ai-economist-gabm
```

3. Install as an editable Python package

```pyfunctiontypecomment
   cd modified-ai-economist-gabm
   pip install -e .
```

## Testing your Install

To test your installation, try running:

```
conda activate modified-ai-economist-gabm
python -c "import modified_ai_economist_gabm"
```