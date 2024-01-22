import boto3
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.llms import bedrock as BedRockLLM
from langchain.agents import Tool
import gradio as gr

duckduckgo_search = DuckDuckGoSearchRun()
bedrock_client = boto3.client(
    service_name='bedrock-runtime', region_name='us-east-1')


def createAgentSetup(niche, location="Brazil"):

    # Define LLM
    llm = BedRockLLM.Bedrock(
        client=bedrock_client, model_id="cohere.command-text-v14"
    )

    # Define Agents

    niche_research_analyst = Agent(
        role="Digital Marketing Niche Research Analyst",
        goal=f"""Analyse the best sub-niches markets of a {niche} niche and suggest based on data such as demand, competition, and profitability in the given {location} the rifht one to choose.
        """,
        backstory=f"""A niche virtuoso, navigates the digital landscape effortlessly. With an innate knack for discovery, he unveils untapped client realms, paving the way for unparalleled success in digital marketing. A silent maestro in the art of {niche} sub-niches revelation.
        """,
        verbose=True,
        allow_delegation=True,
        tools=[duckduckgo_search],
        llm=llm
    )

    icp_expert = Agent(
        role="Digital Makerting ICP (Ideal Customer Profile) Expert",
        goal=f"""Create an ICP of clients in the {niche} niche considering variables as the following: Company Size, Revenue, Decision Makers, Buying Process, Industry, Greography ({location}), Pain Points, Business Goals, Technologies and Attributes
        """,
        backstory=f"""Renowned in digital marketing, a visionary strategist crafts unparalleled Ideal Customer Profiles across diverse niches. Mastermind behind targeted campaigns, their insights redefine customer engagement
        """,
        verbose=True,
        allow_delegation=True,
        llm=llm
    )

    marketing_strategist = Agent(
        role="Digital Marketing Strategy Expert",
        goal=f"""Create the most effecient digital marketing strategy for a client in the given {niche} niche based on {location}
        """,
        backstory=f"""A seasoned Digital Marketing Strategy Expert, boasts a rich history of crafting impactful online campaigns. Armed with a profound understanding of market trends, excel in optimizing digital presence, maximizing conversions, and steering brands towards unparalleled success
        """,
        verbose=True,
        allow_delegation=True,
        llm=llm
    )

    business_consultant = Agent(
        role="Business Consultant",
        goal=f"""Creates an understandable report showing the sub niches, the ICP and the marketing strategy
        """,
        backstory=f"""Seasoned in creating business reports, understands the best strategies on how a SMMA (Social Media Marketing Agency) or a Digital Marketing Company can find and close deals with potencial clients in the {niche} niche
        """,
        verbose=False,
        allow_delegation=True,
        llm=llm
    )

    # Define Tasks

    task1 = Task(
        description=f"""
            Analyse the best sub-niches markets of a {niche} niche and suggest based on data such as demand, competition, and profitability in the given {location} the rifht one to choose
        """,
        agent=niche_research_analyst
    )

    task2 = Task(
        description=f"""
            Create an ICP of clients in the {niche} niche considering variables as the following: Company Size, Revenue, Decision Makers, Buying Process, Industry, Greography ({location}), Pain Points, Business Goals, Technologies and Attributes
        """,
        agent=icp_expert
    )

    task3 = Task(
        description=f"""
            Create the most effecient digital marketing strategy for a client in the given {niche} niche
        """,
        agent=marketing_strategist
    )

    task4 = Task(
        description=f"""Create an understandable report showing the chosen sub niche, the ICP and the marketing strategy
        """,
        agent=business_consultant
    )

    # Create and Run the Crew

    marketing_crew = Crew(
        agents=[niche_research_analyst, icp_expert,
                marketing_strategist, business_consultant],
        tasks=[task1, task2, task3, task4],
        verbose=2,
        process=Process.sequential
    )

    crew_result = marketing_crew.kickoff()
    return crew_result

# Gradio Interface


def run_crewai(niche, location):
    crew_result = createAgentSetup(niche, location)
    return crew_result


iface = gr.Interface(
    fn=run_crewai,
    inputs="text",
    outputs="text",
    title="CrewAI - Marketing Advisors AI Agents",
    description="Enter the niche you want to explore and the location"
)


iface.launch()
