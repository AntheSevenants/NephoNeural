import os
import shutil

from datetime import datetime

from .file_writer import FileWriter
from .helpers import flat_map, unique

class TypeExporter:
    def __init__(self, output_dir, types, skip_novar_tokens=True):
        self.output_dir = output_dir
        self.types = types
        self.skip_novar_tokens = skip_novar_tokens
        
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
            
            # Write lemma.{solution}.cws.tsv file
            self.write_context_solutions(type_inst, type_dir)
            
            # Write lemma.variables.tsv file
            self.write_variables(type_inst, type_dir)
            
    def write_paths_json(self, type_inst, type_dir):
        self.paths = { "models": "{}.models.tsv".format(type_inst.lemma),
                       "solutions": "{}.solutions.tsv".format(type_inst.lemma),
                       "modelsdist": "{}.models.dist.tsv".format(type_inst.lemma),
                       "variables":"{}.variables.tsv".format(type_inst.lemma) }
        
        for dimension_reduction_technique in type_inst.dimension_reduction_techniques:
            self.paths[dimension_reduction_technique.name] = "{}.{}.tsv".format(type_inst.lemma, dimension_reduction_technique.name)
            self.paths[f"{dimension_reduction_technique.name}cws"] = "{}.{}.cws.tsv".format(type_inst.lemma, dimension_reduction_technique.name)
        
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
                if self.skip_novar_tokens:
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

    def write_context_solutions(self, type_inst, type_dir):
        row_template = { "_id": None }
        # By default, all context words are "lost"
        for model_name in type_inst.model_names:
            row_template["{}.x".format(model_name)] = 0
            row_template["{}.y".format(model_name)] = 0

        # Go over each dimension reduction technique that was used for this type
        for dimension_reduction_technique in type_inst.dimension_reduction_techniques:
            rows = [] # will hold the rows for this technique
            for model_name in type_inst.model_names:
                for context_word in type_inst.model_collection.models[model_name].context_words.words:
                    # Get the index of this context word
                    context_word_index = type_inst.model_collection.models[model_name].context_words.words.index(context_word)

                    row = { **row_template,
                            "_id": context_word,
                            f"{model_name}.x": type_inst.model_collection.models[model_name].context_solutions[dimension_reduction_technique.name][context_word_index][0],
                            f"{model_name}.y": type_inst.model_collection.models[model_name].context_solutions[dimension_reduction_technique.name][context_word_index][1]
                          }

                    rows.append(row)

            # Each dimension reduction technique has its own file, so the file for this dimension reduction technique is done
            FileWriter.write("{}{}".format(type_dir, self.paths[f"{dimension_reduction_technique.name}cws"]),
                             rows,
                             content_type="tsv")


    def write_context_solutions_lemma(self, type_inst, type_dir):
        # Go over each dimension reduction technique that was used for this type
        for dimension_reduction_technique in type_inst.dimension_reduction_techniques:
            rows = [] # will hold the rows for this technique

            # Because each cw = one row, we need to amass the cws from all the models first
            all_context_words = list(flat_map(lambda model_name: type_inst.model_collection.models[model_name].context_words.words,
                                              type_inst.model_names))
            # Now, we only want the unique values
            all_context_words = unique(all_context_words)

            # Start building the rows
            for context_word in all_context_words:
                row = { "_id": context_word }
                for model_name in type_inst.model_names:
                    # Get the index of this context word
                    context_word_index = type_inst.model_collection.models[model_name].context_words.words.index(context_word)

                    # If the context word actually appears in this model, use its coordinates
                    if context_word in type_inst.model_collection.models[model_name].context_words.words:
                        row["{}.x".format(model_name)] = type_inst.model_collection.models[model_name].context_solutions[dimension_reduction_technique.name][context_word_index][0]
                        row["{}.y".format(model_name)] = type_inst.model_collection.models[model_name].context_solutions[dimension_reduction_technique.name][context_word_index][1]
                    # Else, the context word is "lost"
                    else:
                        row["{}.x".format(model_name)] = 0
                        row["{}.y".format(model_name)] = 0

                rows.append(row)

            # Each dimension reduction technique has its own file, so the file for this dimension reduction technique is done
            FileWriter.write("{}{}".format(type_dir, self.paths[f"{dimension_reduction_technique.name}cws"]),
                             rows,
                             content_type="tsv")

    def write_variables(self, type_inst, type_dir):
        rows = []

        # "token index" here refers to the index of the token in the list of tokens
        # not the index in the sentence!
        for token_index in range(len(type_inst.token_list)):
            token_id = type_inst.token_ids[token_index]
            sentence = type_inst.sentences[token_index]

            # This is the only way for me to reliably get the correct variable record
            # We can't use ids, because variable order and sentence order might not be the same
            # (it's beyond my control, I don't have the original dataset :( )
            filtered_variables = list(filter(lambda token: token_id == token["_id"], type_inst.variables))

            if len(filtered_variables) == 0:
                print(f"Warning: Skipping {token_id} variables; token does not have associated variables")
                continue

            token = filtered_variables[0]

            # Now, create the raw context
            token = { **token, "_ctxt.raw": self.generate_context(sentence["sentence"],
                                                                  sentence["token_index"]) }

            # Now, add the context words for this token for every model
            for model_name in type_inst.model_names:
                cws_string = ""
                if token_index in type_inst.model_collection.models[model_name].context_words.words_in_sentence:
                    cws_string = ";".join(type_inst.model_collection.models[model_name].context_words.words_in_sentence[token_index])
                    
                token[f"_cws.{model_name}"] = cws_string

            rows.append(token)

        FileWriter.write("{}{}".format(type_dir, self.paths["variables"]),
                         rows,
                         content_type="tsv")

    def generate_context(self, sentence, token_index):
        # We don't want to change the original sentence, so we create a deep copy
        sentence = sentence.copy()

        # Highlight the focus token
        sentence[token_index] = f"<span class='target'>{sentence[token_index]}</span>"

        # Join everything together and return
        return " ".join(sentence)