# Langchain Vision Tools

The Vision Tools library provides a set of tools for image analysis and recognition, leveraging various deep learning models. It includes functionalities for deep image tagging using the DeepDanbooru model, image analysis using the CLIP model, and vision-based predictions using the GPT-4 Vision Preview model.
Installation

Before using the following library please ensure you have:
    - A Supported CUDA version installed (For GPU Support)
    - Visual Studio
    - An OpenAI API Key

To use this library, you need to install the required dependencies. You can install them using the following:

```bash
pip install openai langchain deepdanbooru pillow tensorflow huggingface_hub clip_interrogator python-dotenv
```

Be sure to set OPENAI_API_KEY in your environment
```bash
export OPENAI_API_KEY='api_key'
```

Example Usage
```python
import vision_tools as vt
# Example usage of the VisionTools library
# Create an instance of VisionTools with DeepDanbooru support
vision_tools = vt.VisionTools(is_deepdan=True)

# Image path for analysis
image_path = "path/to/your/image.jpg"

# GPT-4 Vision Preview prediction
prompt = "Describe the image"
gpt_result = vision_tools.gpt_v_predict(prompt, image_path)
print(f"GPT Result: {gpt_result}")

# CLIP Interrogator prediction
clip_result = vision_tools.clip_interrogator_predict(image_path)
print(f"CLIP Result: {clip_result}")

# DeepDanbooru prediction with a score threshold
score_threshold = 0.5
deepdan_result_threshold, deepdan_result_all, deepdan_result_text = vision_tools.deepdan_predict(image_path, score_threshold)
print(f"DeepDanbooru Result (Threshold): {deepdan_result_threshold}")
print(f"DeepDanbooru Result (All): {deepdan_result_all}")
print(f"DeepDanbooru Result (Text): {deepdan_result_text}")

```
Acknowledgements

    OpenAI for the GPT-4 Vision Preview model
    LangChain for the LangChain tools
    DeepDanbooru for the pre-trained model and labels
    TensorFlow for the deep learning framework
    Hugging Face for the CLIP model
    Python Imaging Library (PIL) for image processing
    Python dotenv for environment variable loading
