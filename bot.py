import json
import pytz
import discord
import torch
from transformers import AutoTokenizer, GPTNeoForCausalLM, TextGenerationPipeline

# text prediction

if False: # small model
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
    model = GPTNeoForCausalLM.from_pretrained(
        'EleutherAI/gpt-neo-125M', pad_token_id=tokenizer.eos_token_id)
    # use gpu if available
    device = torch.cuda.current_device() if torch.cuda.is_available() else -1
else: # large model (>10gb)
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
    model = GPTNeoForCausalLM.from_pretrained(
        'EleutherAI/gpt-neo-2.7B', pad_token_id=tokenizer.eos_token_id)
    device = -1 # cpu

pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)

def load_config():
    global config
    with open('bot_config.json') as f:
        config = json.load(f)

load_config()

def get_loss(text):
    inputs = tokenizer(text, return_tensors='pt').to(device)
    outputs = model(**inputs, labels=inputs['input_ids'])
    return outputs.loss.item()

def calc_response_weights():
    # responses should ideally be weighted so that the correct response
    # has the smallest weighted prediction loss

    # current approach is to weight such that the average prediction loss
    # for mismatching example message/response type pairs is 1
    avg_mismatch_losses = []
    for resp1 in config['response_types']:
        avg_mismatch_losses.append(0)
        total_losses = 0
        for resp2 in config['response_types']:
            if resp1 == resp2:
                continue
            for ex in resp2['examples']:
                text = config['intention_template'].format(
                    msg_text=ex, resp=resp1['text']
                )
                avg_mismatch_losses[-1] += get_loss(text)
                total_losses += 1
        avg_mismatch_losses[-1] /= total_losses
        print(f'[{resp1["text"]}] weight: {1 / avg_mismatch_losses[-1]}')

# calc_response_weights()

def gen_text(prompt):
    # compute max_length including prompt
    prompt_len = len(tokenizer(prompt).input_ids)
    return pipeline(
        prompt, max_length=prompt_len + config['pred_length'],
        return_full_text=False, **config['sampling_args']
    )[0]['generated_text']

def get_intention(msg_text):
    # (doesn't work very well with current templates)
    # todo: process prompt once and only compute loss for response types
    resp_losses = []
    for resp in config['response_types']:
        text = config['intention_template'].format(
            msg_text=msg_text, resp=resp['text']
        )
        resp_losses.append((resp['text'], get_loss(text) * resp['weight']))
    
    # get action with lowest prediction loss
    print(sorted(resp_losses, key=lambda v: v[1]))
    return sorted(resp_losses, key=lambda v: v[1])[0][0]


# bot

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

@client.event
async def on_message(message):
    # don't respond to own messages
    if message.author == client.user:
        return
    
    # if bot is mentioned
    if client.user in message.mentions:
        async with message.channel.typing():
            bot_name = message.guild.me.display_name
            msg_text = message.clean_content.replace(f'@{bot_name}', '').strip()
            print(f'---\n\n[Message]: {msg_text}\n')

            # reload config
            if msg_text == '.config':
                load_config()
                await message.channel.send('Reloaded config')
                return
            
            template = config['message_template']
            prompt_prefix = template['prefix'].format(
                bot_name=bot_name
            )
            
            messages = reversed([msg async for msg in message.channel.history(
                limit=config['message_history']
            )])
            for msg in messages:
                msg_name = msg.author.display_name
                # todo: use '%-I' if not running on Windows
                msg_time = msg.created_at.astimezone(
                    pytz.timezone('EST')
                ).strftime('%#I:%M %p')
                # remove bot mention
                msg_text = msg.clean_content.replace(f'@{bot_name}', '').strip()

                prompt_prefix += template['message'].format(
                    msg_name=msg_name, msg_time=msg_time, msg_text=msg_text
                )
            
            # generate response
            resp = ''
            for _ in range(config['max_num_preds']):
                prompt = prompt_prefix + template['postfix'].format(
                    bot_name=bot_name, msg_time=msg_time, resp=resp
                )
                # continue from template
                print(f'[Prompt]: {prompt}\n')
                pred = gen_text(prompt)
                print(f'[Prediction]: {pred}\n')
                pred_parts = pred.split('\n')
                resp += pred_parts[0]
                # stop if new line generated
                if len(pred_parts) > 1:
                    break
        
        # send response
        if not resp.strip():
            resp = config['empty_message_replacement']
        await message.channel.send(resp)


with open('bot.token') as f:
    token = f.read()
client.run(token)