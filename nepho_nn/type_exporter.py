import os
import shutil

from datetime import datetime

from .file_writer import FileWriter

class TypeExporter:
    def __init__(self, output_dir, types):
        self.output_dir = output_dir
        self.types = types
        
        # Append a trailing slash to the path given so we're sure it's a directory
        if (self.output_dir[-1] != "/"):
            self.output_dir = self.output_dir + "/"
                        
    def export(self):
        # Delete existing export
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        
        # Create directory anew
        os.makedirs(self.output_dir)
        
        self.write_euclidean_register()
        self.write_type_data()
        
    # Writes the euclidean register to the root of the output directory
    def write_euclidean_register(self):
        euclidean_register_path = "{}euclidean_register.tsv".format(self.output_dir)
        
        # Will hold dicts of data for each row. Serves as basis for the dataframe
        rows = []
        
        # Holds today's date in YYYY-MM-DD
        today_date = datetime.today().strftime('%Y-%m-%d')
        
        # For each type we're interested in, create a data row
        for type_inst in self.types:
            row = { "type": type_inst.lemma,
                    "models": len(type_inst.model_names),
                    "stress": 0, # ???
                    "date": today_date,
                    "part_of_speech": "miep" # todo
                  }
            rows.append(row)
        
        FileWriter.write(euclidean_register_path, rows, content_type="tsv")
        
    def write_type_data(self):
        for type_inst in self.types:
            type_dir = "{}{}/".format(self.output_dir, type_inst.lemma)
            
            # Create a directory for each type
            os.makedirs(type_dir)
            
            # Write the paths.json file
            self.write_paths_json(type_inst, type_dir)
            
            # Write lemma.models.tsv file
            self.write_models(type_inst, type_dir)

            # Write lemma.models.dist.tsv file
            self.write_model_distances(type_inst, type_dir)
            
            # Write lemma.{solution}.tsv file
            self.write_solutions(type_inst, type_dir)
            
            # Write lemma.variables.tsv file
            self.write_variables(type_inst, type_dir)
            
    def write_paths_json(self, type_inst, type_dir):
        self.paths = { "models": "{}.models.tsv".format(type_inst.lemma),
                       "solutions": "{}.solutions.tsv".format(type_inst.lemma),
                       "modelsdist": "{}.models.dist.tsv".format(type_inst.lemma),
                       "variables":"{}.variables.tsv".format(type_inst.lemma) }
        
        for dimension_reduction_technique in type_inst.dimension_reduction_techniques:
            self.paths[dimension_reduction_technique.name] = "{}.{}.tsv".format(type_inst.lemma, dimension_reduction_technique.name)
        
        FileWriter.write("{}paths.json".format(type_dir), self.paths, content_type="json")
        
    def write_models(self, type_inst, type_dir):
        models_json_path = "{}{}".format(type_dir, self.paths["models"])
        
        rows = []
        
        for model_name in type_inst.model_names:
            model_index = type_inst.model_names.index(model_name)
            
            # For the model coordinates, we just pick the first solution available
            # TODO: make this configurable
            chosen_solution = list(type_inst.solutions.keys())[0]

            row = { "_model": model_name,
                    "model.x": type_inst.solutions[chosen_solution][model_index][0],
                    "model.y": type_inst.solutions[chosen_solution][model_index][1],
                    "foc_model_type": type_inst.model_collection.models[model_name].metadata["architecture"],
                    "foc_layer": type_inst.model_collection.models[model_name].metadata["layer"]
                  }
            
            rows.append(row)
        
        FileWriter.write(models_json_path, rows, content_type="tsv")

    def write_model_distances(self, type_inst, type_dir):
        # We look at the giant distance matrix, and then turn it into the NephoVis-compatible format
        rows = []

        # Loop over all models (outer)
        # = SOURCE for the distance matrix
        for model_name in type_inst.model_names:
            row = { "_model": model_name,
                    **type_inst.model_collection.models[model_name].model_similarity_vector }

            rows.append(row)

        FileWriter.write("{}{}".format(type_dir, self.paths["modelsdist"]),
                             rows,
                             content_type="tsv")
        
    def write_medoids(self, type_inst, type_dir):
        pass
    
    def write_solutions(self, type_inst, type_dir):       
        # Content of solutions.json:
        solutions = {}
        
        # Go over each dimension reduction technique that was used for this type
        for dimension_reduction_technique in type_inst.dimension_reduction_techniques:
            # Add the name of this dimension reduction technique to the overview of techniques
            solutions[dimension_reduction_technique.name] = dimension_reduction_technique.name
            
            rows = [] # will hold the rows for this technique
            
            for token_index in range(len(type_inst.token_list)):
                token_id = type_inst.token_ids[token_index]

                # This token id does not have variables associated with it
                # So we won't add it to the dataset (crash stopgap)
                if not token_id in type_inst.token_ids_variables_available:
                    print(f"Warning: Skipping {token_id}; token does not have associated variables")
                    continue

                row = { "_id": token_id }
                for model_name in type_inst.model_names:
                    row["{}.x".format(model_name)] = type_inst.model_collection.models[model_name].solutions[dimension_reduction_technique.name][token_index][0]
                    row["{}.y".format(model_name)] = type_inst.model_collection.models[model_name].solutions[dimension_reduction_technique.name][token_index][1]
                        
                rows.append(row)
            
            # Each dimension reduction technique has its own file, so the file for this dimension reduction technique is done
            FileWriter.write("{}{}".format(type_dir, self.paths[dimension_reduction_technique.name]),
                             rows,
                             content_type="tsv")
            
        solutions_json_path = "{}{}".format(type_dir, self.paths["solutions"])
        FileWriter.write(solutions_json_path, solutions, content_type="json")

    def write_variables(self, type_inst, type_dir):
        rows = []

        for token in type_inst.variables:
            rows.append(token)

        FileWriter.write("{}{}".format(type_dir, self.paths["variables"]),
                         rows,
                         content_type="tsv")