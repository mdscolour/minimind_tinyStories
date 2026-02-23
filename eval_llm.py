import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind Model Inference and Dialogue")
    parser.add_argument('--load_from', default='model', type=str, help="Model load path（model=nativetorch，other paths=transformers）")
    parser.add_argument('--save_dir', default='out', type=str, help="Model weight directory")
    parser.add_argument('--weight', default='full_sft', type=str, help="Weight name prefix (pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo)")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA weight name (None=not use, options: lora_identity, lora_medical)")
    parser.add_argument('--hidden_size', default=512, type=int, help="Hidden layer dimension（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Number of hidden layers（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="yesnouseMoEarchitecture（0=no，1=yes）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="Enable RoPE position encoding extrapolation (4x, only for position encoding)")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="Maximum generation length (note: not actual long text capability)")
    parser.add_argument('--temperature', default=0.85, type=float, help="Generation temperature, controls randomness (0-1, higher=more random)")
    parser.add_argument('--top_p', default=0.85, type=float, help="Nucleus sampling threshold (0-1)")
    parser.add_argument('--historys', default=0, type=int, help="Number of history turns to include (must be even, 0=no history)")
    parser.add_argument('--show_speed', default=1, type=int, help="Show decode speed (tokens/s)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="Device")
    args = parser.parse_args()
    
    prompts = [
        # 1 Easiest: single character + single simple event 
        "Tell a short story for kids: Mimi the little cat could not find her mother.",
    
        # 2 Adds emotional change (fear → comfort)
        "Tell a short story for kids: Tim was scared to sleep alone for the first time.",
    
        # 3 Two characters and interaction (consistency required)
        "Tell a story for kids: Anna and her brother Ben found a lost puppy in the park.",
    
        # 4 Problem solving (needs causal reasoning and coherence)
        "Continue the story for kids: Lily noticed the class plant was dying and no one knew why.",
    
        # 5 Full narrative arc (mistake → guilt → apology → happy ending → moral)
        "Finish a children's story with a happy ending: Max broke his friend's favorite toy and felt very bad.",
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] test\n[1] \n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('User: '), '')
    for prompt in prompt_iter:
        setup_seed(2026) # or setup_seed(random.randint(0, 2048))
        if input_mode == 0: print(f'User: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        if args.weight == 'reason': templates["enable_thinking"] = True # Only for Reason model
        inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + prompt)
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('Model: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()