import asyncio
import math
from io import BytesIO
import base64
import re
import json
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

MAX_PIXELS = 1350 * 28 * 28
AGENT = r"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
click(start_box='[x1, y1, x2, y2]')
left_double(start_box='[x1, y1, x2, y2]')
right_single(start_box='[x1, y1, x2, y2]')
drag(start_box='[x1, y1, x2, y2]', end_box='[x3, y3, x4, y4]')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='[x1, y1, x2, y2]', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.

## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
"""

class UITars():
    def __init__(self):
        model_id = 'bytedance-research/UI-TARS-7B-DPO'
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype="auto", device_map="auto", 
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )

    def inference(self, image, instruction): 
        image = Image.fromarray(image)
        scale = MAX_PIXELS / image.width / image.height
        if scale < 1:
            scale = math.sqrt(scale)
            image = image.resize((image.width * scale, image.height * scale))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        encoded_string = base64.b64encode(buffer.read()).decode('utf-8')
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": AGENT + instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        ).to("cuda")
        output_ids = self.model.generate(**inputs, temperature=0.01, top_p=0.7, do_sample=True, max_new_tokens=1000)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        generated_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        answer = self.process_text(generated_texts[0])
        return json.dumps(answer, separators=(',', ':'))
    
    def parse_action(self, action_str: str):
        function_pattern = r"^(\w+)\((.*)\)$"
        match = re.match(function_pattern, action_str.strip())
        if not match:
            return None
        function_name, args_str = match.groups()
        kwargs = {}
        if args_str.strip():
            arg_pairs = re.findall(r"(?:[^,']|'[^']*')+", args_str) or []
            for pair in arg_pairs:
                parts = pair.split('=')
                if len(parts) < 2:
                    continue
                key = parts[0].strip()
                value = '='.join(parts[1:]).strip().strip('\'\"')
                kwargs[key] = value
        return {
            'function': function_name,
            'args': kwargs,
        }
    
    def process_text(self, text):
        thought = ''
        reflection = None
        action_str = ''
        if text.startswith('Thought:'):
            thought_match = re.search(r'Thought: ([\s\S]+?)(?=\s*Action:|$)', text)
            if thought_match:
                thought = thought_match.group(1).strip()
        elif text.startswith('Reflection:'):
            reflection_match = re.search(r'Reflection: ([\s\S]+?)Action_Summary: ([\s\S]+?)(?=\s*Action:|$)', text)
            if reflection_match:
                reflection = reflection_match.group(1).strip()
                thought = reflection_match.group(2).strip()
        elif text.startswith('Action_Summary:'):
            summary_match = re.search(r'Action_Summary: (.+?)(?=\s*Action:|$)', text)
            if summary_match:
                thought = summary_match.group(1).strip()
        if 'Action:' not in text:
            action_str = text
        else:
            action_parts = text.split('Action:')
            action_str = action_parts[-1]

        all_actions = action_str.split('\n\n')
        actions = []

        for raw_str in all_actions:
            action_instance = self.parse_action(raw_str.replace('\n', '\\n').lstrip())
            action_type = ''
            action_inputs = {}

            if action_instance:
                action_type = action_instance['function']
                params = action_instance['args']
                action_inputs = {}

                for param_name, param in params.items():
                    if not param:
                        continue
                    trimmed_param = param.strip()
                    action_inputs[param_name.strip()] = trimmed_param

                    if 'start_box' in param_name or 'end_box' in param_name:
                        numbers = trimmed_param.replace('[', '').replace(']', '').replace('(', '').replace(')', '').split(',')
                        float_numbers = [float(num) / 1000 for num in numbers]
                        if len(float_numbers) == 2:
                            float_numbers.extend([float_numbers[0], float_numbers[1]])
                        action_inputs[param_name.strip()] = json.dumps(float_numbers)
            actions.append({
                'reflection': reflection,
                'thought': thought,
                'action_type': action_type,
                'action_inputs': action_inputs,
            })
        return actions

async def start():
    uitars = UITars()
    image = Image.open('screenshot.png')
    answer = uitars.inference(image, 'click my computer')
    print(answer)

if __name__ == "__main__":
    asyncio.run(start())
