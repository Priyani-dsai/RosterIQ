import json


class ProceduralMemory:

    def __init__(self, file_path="memory/procedures.json"):

        self.file_path = file_path

        try:
            with open(file_path, "r") as f:
                self.procedures = json.load(f)
        except:
            self.procedures = {}


    def add_procedure(self, name, description, steps):

        self.procedures[name] = {
            "description": description,
            "steps": steps
        }

        self._save()


    def get_procedure(self, name):

        return self.procedures.get(name, None)


    def list_procedures(self):

        return list(self.procedures.keys())


    def _save(self):

        with open(self.file_path, "w") as f:
            json.dump(self.procedures, f, indent=4)