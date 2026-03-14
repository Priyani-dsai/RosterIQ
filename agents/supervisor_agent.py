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

        # -----------------------------
        # HARD RULE ROUTING
        # -----------------------------

        # Visualization trigger (higher priority)
        if any(word in query_lower for word in ["chart", "plot", "visualize","visualization", "graph", "distribution","histogram"]):
           return "visualization_analysis"

        # Web search trigger
        if any(word in query_lower for word in ["search","web","lookup","look up","online"]):
           return "external_research"
        # -----------------------------
        # LLM ROUTING
        # -----------------------------

        procedures = self.procedural_memory.procedures
        procedure_names = list(procedures.keys())

        procedure_descriptions = ""

        for name, p in procedures.items():
            procedure_descriptions += f"{name}: {p['description']}\n"

        routing_prompt = f"""
You are a supervisor agent for a provider roster pipeline analytics system.

User query:
{query}

Available procedures:
{procedure_descriptions}

Return ONLY one of the following procedure names:

{", ".join(procedure_names)}

Do NOT explain anything.
Return ONLY the procedure name.
"""

        decision = self.llm.invoke(routing_prompt).content.strip()

        # clean LLM output
        decision = decision.replace(".", "").replace(",", "").strip()

        # safety fallback
        if decision not in procedure_names:
            decision = "pipeline_health_check"

        return decision


    def route_query(self, query):

        # Step 1: choose procedure
        procedure_name = self.choose_procedure(query)

        print("Selected procedure:", procedure_name)

        procedure = self.procedural_memory.get_procedure(procedure_name)

        # fallback if something unexpected happens
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

        # Extract final answer
        answer = result.get("output", "")

        # Clean reasoning traces
        if "Final Answer:" in answer:
            answer = answer.split("Final Answer:")[-1].strip()

        # remove ReAct artifacts
        for tag in ["Thought:", "Action:", "Action Input:", "Observation:"]:
            if tag in answer:
                answer = answer.split(tag)[0].strip()

        if not answer:
            answer = "The agent completed the analysis but did not produce a formatted response."

        return {
            "procedure": procedure_name,
            "agent": agent_used,
            "output": answer
        }