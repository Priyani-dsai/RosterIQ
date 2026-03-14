import os
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq

from agents.pipeline_agent import create_pipeline_agent
from agents.quality_agent import create_quality_agent
from memory.procedural_memory import ProceduralMemory

class SupervisorAgent:

    def __init__(self, roster_df):

        # Specialized agents
        self.pipeline_agent = create_pipeline_agent(roster_df)
        self.quality_agent = create_quality_agent(roster_df)

        # Procedural memory
        self.procedural_memory = ProceduralMemory()

        # LLM for decision making
        self.llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
        )


    def choose_procedure(self, query):

        query_lower = query.lower()

        # ---- HARD RULE ROUTING ----

        # Web search trigger
        if any(word in query_lower for word in ["search","web","lookup","look up","online"]):
            return "external_research"

        # Visualization trigger
        if any(word in query_lower for word in ["chart", "plot", "visualize", "graph", "distribution"]):
            return "visualization_analysis"

        # ---- LLM ROUTING ----

        procedures = self.procedural_memory.procedures

        procedure_descriptions = ""
        procedure_names = list(procedures.keys())

        for name, p in procedures.items():
            procedure_descriptions += f"{name}: {p['description']}\n"

        routing_prompt = f"""
You are a supervisor agent for a provider roster pipeline analytics system.

User query:
{query}

Available procedures:
{procedure_descriptions}

You MUST return ONLY one of these procedure names:

{", ".join(procedure_names)}

Return ONLY the procedure name.
Do not explain anything.
"""

        decision = self.llm.invoke(routing_prompt).content.strip()

        # Safety filter
        if decision not in procedure_names:
            decision = procedure_names[0]

        return decision

    def route_query(self, query):

        # Step 1: choose procedure
        procedure_name = self.choose_procedure(query)

        print("Selected procedure:", procedure_name)

        procedure = self.procedural_memory.get_procedure(procedure_name)

        # fallback if LLM returns something unexpected
        if not procedure:
            procedure = self.procedural_memory.get_procedure("pipeline_health_check")

        agent_type = procedure["agent"]

        # Step 2: route to correct agent
        if agent_type == "pipeline_agent":
            agent_used = "Pipeline Health Agent"
            result = self.pipeline_agent.invoke({"input": query})
        else:
            agent_used = "Data Quality Agent"
            result = self.quality_agent.invoke({"input": query})

        # Extract final answer only
        answer = result.get("output", "")

        # Clean LangChain reasoning if present
        if "Final Answer:" in answer:
            answer = answer.split("Final Answer:")[-1].strip()

        return {
        "procedure": procedure_name,
        "agent": agent_used,
        "output": answer
    }