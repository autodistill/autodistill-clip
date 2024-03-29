<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill CLIP Module

This repository contains the code supporting the CLIP base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[CLIP](https://github.com/openai/CLIP), developed by OpenAI, is a computer vision model trained using pairs of images and text. You can use CLIP with autodistill for image classification.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [CLIP Autodistill documentation](https://autodistill.github.io/autodistill/base_models/clip/).

## Installation

To use CLIP with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-clip
```

## Quickstart

```python
from autodistill_clip import CLIP
from autodistill.detection import CaptionOntology

# define an ontology to map class names to our CLIP prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = CLIP(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)

results = base_model.predict("./context_images/test.jpg")

print(results)

base_model.label("./context_images", extension=".jpeg")
```

## License

The code in this repository is licensed under an [MIT license](LICENSE.md).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!