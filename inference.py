from llama.tokenizer import Tokenizer
from llama.model import ModelArgs, Llama
import torch


def inference():
    torch.manual_seed(1)

    tokenizer_path = "/project/saifhash_1190/llama2-7b/tokenizer.model"
    model_path = "/project/saifhash_1190/llama2-7b/consolidated.00.pth"
    lora_path = "/home1/xcai6647/lora_weight.pth"

    tokenizer = Tokenizer(tokenizer_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    lora = torch.load(lora_path, map_location="cpu")

    model_args = ModelArgs()
    torch.set_default_tensor_type(torch.cuda.HalfTensor)  # load model in fp16
    model = Llama(model_args)
    model.load_state_dict(checkpoint, strict=False)
    model.load_state_dict(lora, strict=False)
    model.to("cuda")

    # prompts = [
    #     # For these prompts, the expected answer is the natural continuation of the prompt
    #     "I believe the meaning of life is",
    #     "Simply put, the theory of relativity states that ",
    #     """A brief message congratulating the team on the launch:

    #     Hi everyone,

    #     I just """,
    #     # Few shot prompt (providing a few examples before asking model to complete more);
    #     """Translate English to French:

    #     sea otter => loutre de mer
    #     peppermint => menthe poivrÃ©e
    #     plush girafe => girafe peluche
    #     cheese =>""",
    # ]

    def Alpaca_prompt(instruction, input):
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. 
        Write a response that appropriately completes the request.
        ### Instruction:{instruction}
        ### Input:{input}
        ### Response:"""

    prompts = [
        Alpaca_prompt("Convert the following sentence into the present continuous tense",
                      "He reads books"),
        Alpaca_prompt("Give an example of a metaphor that uses the following object",
                      "Stars"),
        Alpaca_prompt("Describe the following person",
                      "John"),
        Alpaca_prompt("Construct an argument to defend the following statement.",
                      "Alternative energy sources are critical to solving the climate crisis"),
    ]

    model.eval()
    results = model.generate(tokenizer, prompts, max_gen_len=64, temperature=0.6, top_p=0.9)

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    inference()