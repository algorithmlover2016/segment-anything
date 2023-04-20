from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import pytest
import cv2

model_weights = {
        "vit_h" : "../weights/sam_vit_h_4b8939.pth",
        "vit_b" : "../weights/sam_vit_b_01ec64.pth",
        "vit_l" : "../weights/sam_vit_l_0b3195.pth",
}
image3 = cv2.imread("./testImgs/3.jpg")
image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
imagewall2 = cv2.imread("./testImgs/wall2.jpg")
imagewall2 = cv2.cvtColor(imagewall2, cv2.COLOR_BGR2RGB)

model_type = 'vit_l'
mask_generatorL = SamAutomaticMaskGenerator(sam_model_registry[model_type](checkpoint=model_weights[model_type]).to("cuda"))

model_type = 'vit_b'
# masks = mask_generatorL.generate(image3)
mask_generatorB = SamAutomaticMaskGenerator(sam_model_registry[model_type](checkpoint=model_weights[model_type]).to("cuda"))
# masks = mask_generatorB.generate(image3)
model_type = 'vit_h'
mask_generatorH = SamAutomaticMaskGenerator(sam_model_registry[model_type](checkpoint=model_weights[model_type]).to("cuda"))
# masks = mask_generatorH.generate(image3)


def sam_modelsL():
    masks = mask_generatorL.generate(image3)
    masks = mask_generatorL.generate(imagewall2)

def sam_modelsB(model_type = "vit_b"):
    masks = mask_generatorB.generate(image3)
    masks = mask_generatorB.generate(imagewall2)

def sam_modelsH(model_type = "vit_h"):
    masks = mask_generatorH.generate(image3)
    masks = mask_generatorH.generate(imagewall2)

def test_modelsH(benchmark):
    benchmark(sam_modelsH)

def test_modelsL(benchmark):
    benchmark(sam_modelsL)

def test_modelsB(benchmark):
    benchmark(sam_modelsB)

#  pytest -q sam_test.py &> log_cuda.txt 2>&1 &
