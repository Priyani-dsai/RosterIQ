import json


class ProceduralMemory:

    def __init__(self, file_path="memory/procedures.json"):

        self.file_path = file_path

        try:
            with open(file_path, "r") as f:
                self.procedures = json.load(f)
        except:
            self.procedures = {}


    def add_procedure(self, name, description, agent, steps, tools):

        self.procedures[name] = {
        "description": description,
        "agent": agent,
        "steps": steps,
        "tools": tools
        }

        self._save()


    def get_procedure(self, name):

        proc = self.procedures.get(name, None)

        print("Procedural memory retrieved:", name)

        return proc


    def list_procedures(self):

        return list(self.procedures.keys())


    def _save(self):

        with open(self.file_path, "w") as f:
            json.dump(self.procedures, f, indent=4)