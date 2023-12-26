# Intergrated Annotation Tool

## API

### Request Body

- `task`: `detect`, `segmentation`

```json
req = {
    "images": [
        {
            "name": "image's name or frame number",
            "image": "encoded image by base64",
            "shape": "shape of image(WHC)"
        },
        {
            "...": "..."
        }
    ],
    "types": "detect or segmentation",
    "classes": ["class number", "class number", "..."],
    "model": "model name, model size or path to model",
    "base_color": "None if task is not clustering"
}
```

- `task`: `pose`

```json
req = {
    "images": [
        {
            "name": "image's name or frame number",
            "image": "encoded image by base64",
            "shape": "shape of image(WHC)"
        },
        {
            "...": "..."
        }
    ],
    "types": "pose",
    "classes": ["None if task is pose or clustering"],
    "model": "model name, model size or path to model",
    "base_color": ["None if task is not clustsering"]
}
```

- `task`: `clustering`

```json
req = {
    "images": [
        {
            "name": "image's name or frame number",
            "image": "encoded image by base64",
            "shape": "shape of image(WHC)"
        },
        {
            "...": "..."
        }
    ],
    "types": "clustering",
    "classes": ["None if task is pose of clustering"],
    "model": "None if task is clustering",
    "base_color": ["values of BGR in list or tuple"]
}
```
