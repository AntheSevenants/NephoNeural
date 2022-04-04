import json
import pandas as pd

class FileWriter:
    def write(path, content, content_type=False):
        if content_type == "json":
            content = json.dumps(content)
        elif content_type == "tsv":
             # Create a dataframe from the rows of data
            df = pd.DataFrame(content)
            # Write the data to a TSV
            df.to_csv(path, sep="\t", index=False)
            return
        
        with open(path, "wt") as writer:
            writer.write(content)