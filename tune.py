import os
import torch
from typing import Dict, List
from datasets import load_dataset, concatenate_datasets, Dataset
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from peft import LoraConfig, PeftConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, GPT2LMHeadModel
from trl import SFTTrainer


def get_rename_pairs(dataset_name: str) -> List[Dict[str, str]]:
    """takes a dataset's name, loads the columns to be renamed from stored map, returns pairs to be renamed for that dataset"""
    rename_map = {
        'samsum': [
            {
                'original': 'dialogue',
                'rename': 'text'
            },
        ],
    }
    # ternary operator for the cases with no renames
    return rename_map[dataset_name] if dataset_name in rename_map else []


def rename_columns(ds: Dataset, rename_pairs: List[Dict[str, str]]) -> Dataset:
    """renames the columns of a dataset based on rename_pairs"""
    for p in rename_pairs:
        ds = ds.rename_column(p['original'], p['rename'])
    return ds


def get_hf_datasets():
    """Returns names of the datasets"""
    return ['samsum', 'billsum']


def get_field_standardization_dicts() -> List[Dict[str, str | List[str] | PromptTemplate]]:
    """returns old field, new field, fields for prompt, and the prompt for each standardization"""
    return [
        {
            'old_field': 'summary',
            'new_field': 'standardized_summary',
            'prompt_fields': ['summary'],
            'template': "Reword, if necessary, the following summary in the tone of a factual documentation writer. Hone in on main points and keep structure simple: {summary}"
        },
    ]


def get_data(hf_dataset_names: List[str] = [], csv_dataset_names: List[str] = []):
    """retrieves data from the hugging face datasets and csv files and renames it to proper format"""
    datasets = []
    for d in hf_dataset_names:
        ds = load_dataset(d)
        ds = rename_columns(ds, get_rename_pairs(d))
        datasets.append(ds['train'].select(range(10)))
    for d in csv_dataset_names:
        ds = load_dataset('csv', data_files=d)
        ds = rename_columns(ds, get_rename_pairs(d))
        datasets.append(ds['train'].select(range(10)))

    return concatenate_datasets(datasets)


def field_standardization(raw_data: Dataset, template: PromptTemplate, input_variables: List[str], field: str) -> List[str]:
    """runs the raw data through a template (with input variables) to standardize a field using GPT-4 calls"""
    llm = OpenAI(temperature=.1)
    prompt = PromptTemplate(
        input_variables=input_variables,
        template=template,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    standardized = [chain.run(s[field]).strip() for s in raw_data]
    return standardized


def get_standardized_data(field_dicts: List[Dict[str, str | List[str]]] = [], hf_dataset_names: List[str] = [], csv_dataset_names: List[str] = []) -> Dataset:
    """returns dataset with prompt-standardized GPT-4 `old_field` """
    summary_data = get_data(hf_dataset_names=hf_dataset_names,
                            csv_dataset_names=csv_dataset_names)
    for f_d in field_dicts:
        standardized_summaries = field_standardization(
            raw_data=summary_data,
            template=f_d['template'],
            input_variables=f_d['prompt_fields'],
            field=f_d['old_field']
        )
        summary_data = summary_data.add_column(
            f_d['new_field'], standardized_summaries)
    return summary_data


def quantize_and_load_model(model_name: str = "meta-llama/Llama-2-70b-hf") -> AutoModelForCausalLM:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    device_map = {"": 0}

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        use_auth_token=True
    )

    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    return base_model


def get_lora_config():
    return LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )


def get_tokenizer(model_name: str = "meta-llama/Llama-2-70b-hf"):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def get_training_args():
    output_dir = "./results"
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=500
    )


def get_trainer(model: AutoModelForCausalLM, dataset: Dataset, peft_config: PeftConfig, tokenizer: AutoTokenizer) -> SFTTrainer:
    training_args = get_training_args()
    return SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )


def print_fields(data: Dataset, fields: List[str]):
    for s in data:
        for f in fields:
            print(s[f])
            print()
        print('\n-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    return


def get_output_file(output_dir: str = './results') -> str:
    return os.path.join(output_dir, "final_checkpoint")


def main():
    # sets the device to use Metal Performance Shaders for all uses of PyTorch in the file
    torch.set_default_device('mps')
    # retrieves api key from SPANNING_OPEN_API_KEY and adds it to environment as OPENAI_API_KEY for Open AI method calls
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    # sets name of the model, quantizes the model, and loads the quantized model into base_model
    HF_MODEL_NAME = "meta-llama/Llama-2-70b-hf"
    base_model = quantize_and_load_model(HF_MODEL_NAME)

    # retrieves the data
    hf_dataset_names = get_hf_datasets()
    field_standardization_dicts = get_field_standardization_dicts()
    summary_data = get_standardized_data(
        field_dicts=field_standardization_dicts, hf_dataset_names=hf_dataset_names)

    # gets the LoRA config and tokenizer and creates the trainer for adaptation
    lora_config = get_lora_config()
    tokenizer = get_tokenizer(HF_MODEL_NAME)
    trainer = get_trainer(base_model, summary_data, lora_config, tokenizer)
    trainer.train()

    output_file = get_output_file()
    trainer.model.save_pretrained(output_file)

    print_fields(summary_data, ['summary', 'standardized_summary'])


main()