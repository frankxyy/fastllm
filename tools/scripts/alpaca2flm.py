import sys
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastllm_pytools import torch2flm

if __name__ == "__main__":
    exportPath = sys.argv[1] if (sys.argv[1] is not None) else "llama-fp32.flm";
    # tokenizer = LlamaTokenizer.from_pretrained('/home/xuyangyang/13B_v2/checkpoint-49200');
    # model = LlamaForCausalLM.from_pretrained('/home/xuyangyang/13B_v2/checkpoint-49200').half();

    tokenizer = AutoTokenizer.from_pretrained('/home/xuyangyang/7B_converted/');
    model = AutoModelForCausalLM.from_pretrained('/home/xuyangyang/7B_converted/').float();

    model = model.eval()

    # tokenizer = LlamaTokenizer.from_pretrained('/workspace/13B_v2/checkpoint-49200');
    # model = LlamaForCausalLM.from_pretrained('/workspace/13B_v2/checkpoint-49200').half();

    torch2flm.tofile(exportPath, model, tokenizer);
