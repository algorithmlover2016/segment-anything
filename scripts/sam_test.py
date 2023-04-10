from segment_anything import build_sam, SamAutomaticMaskGenerator, build_sam_vit_l, build_sam_vit_b
import pytest
import cv2

model_weights = {
        "H" : "../weights/sam_vit_h_4b8939.pth",
        "B" : "../weights/sam_vit_b_01ec64.pth",
        "L" : "../weights/sam_vit_l_0b3195.pth",
}
image3 = cv2.imread("./testImgs/3.jpg")
image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
imagewall2 = cv2.imread("./testImgs/wall2.jpg")
imagewall2 = cv2.cvtColor(imagewall2, cv2.COLOR_BGR2RGB)

mask_generatorL = SamAutomaticMaskGenerator(build_sam_vit_l(checkpoint=model_weights['L']).to("cuda"))
# masks = mask_generatorL.generate(image3)
mask_generatorB = SamAutomaticMaskGenerator(build_sam_vit_b(checkpoint=model_weights['B']).to("cuda"))
# masks = mask_generatorB.generate(image3)
mask_generatorH = SamAutomaticMaskGenerator(build_sam(checkpoint=model_weights['H']).to("cuda"))
# masks = mask_generatorH.generate(image3)


def sam_modelsL():
    masks = mask_generatorL.generate(image3)
    masks = mask_generatorL.generate(imagewall2)

def sam_modelsB(model_type = "B"):
    masks = mask_generatorB.generate(image3)
    masks = mask_generatorB.generate(imagewall2)

def sam_modelsH(model_type = "H"):
    masks = mask_generatorH.generate(image3)
    masks = mask_generatorH.generate(imagewall2)

def test_modelsH(benchmark):
    benchmark(sam_modelsH)

def test_modelsL(benchmark):
    benchmark(sam_modelsL)

def test_modelsB(benchmark):
    benchmark(sam_modelsB)

