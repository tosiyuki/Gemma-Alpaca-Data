import json
from pathlib import Path


if __name__ == "__main__":
    with Path("./regen.json").open() as f:
        gen_datas = json.load(f)

    upload_datas = []
    for gen_data in gen_datas:
        # data cleaning
        if gen_data["input"] in ["<No input>", "No input", "No input required", "N/A", "None", "None."]:
            gen_data["input"] = ""
        if gen_data["output"] == "":
            continue

        upload_data = {}
        upload_data["instruction"] = gen_data["instruction"]
        upload_data["input"] = gen_data["input"]
        upload_data["output"] = gen_data["output"]
        upload_datas.append(upload_data)

    with Path(f"./gemma_alpaca_data_{int(len(upload_datas)/1000)}k.json").open("w") as f:
        json.dump(upload_datas, f, indent=2, ensure_ascii=False)
