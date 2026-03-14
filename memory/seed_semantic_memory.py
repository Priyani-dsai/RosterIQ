from semantic_memory import SemanticMemory

memory = SemanticMemory()

knowledge = [

"PRE_PROCESSING failures usually indicate malformed provider data.",
"INGESTION failures often occur due to mismatched source system schemas.",
"AvailityPDM source system frequently causes ingestion delays.",
"ProviderGroup source system sometimes produces incomplete provider records.",
"DART_GENERATION delays often originate from earlier ingestion failures."

]

for k in knowledge:
    memory.add_knowledge(k)

print("Semantic knowledge stored.")