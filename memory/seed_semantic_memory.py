from memory.semantic_memory import SemanticMemory


memory = SemanticMemory()

knowledge = [

"PRE_PROCESSING failures usually indicate malformed provider data.",
"INGESTION failures often occur due to mismatched source system schemas.",
"AvailityPDM source system frequently causes ingestion delays.",
"ProviderGroup source system sometimes produces incomplete provider records.",
"DART_GENERATION delays often originate from earlier ingestion failures.",
"DART_REVIEW failures usually indicate validation errors.",
"PRE_PROCESSING delays may occur due to large provider files.",
"High DART_GEN_DURATION indicates transformation bottlenecks.",
"Schema mismatches frequently cause ingestion failures.",
"Incomplete provider records cause validation failures."

]

for k in knowledge:
    memory.add_knowledge(k)

print("Semantic knowledge stored in vector database.")