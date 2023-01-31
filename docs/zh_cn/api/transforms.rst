.. role:: hidden
    :class: hidden-section

mmocr.datasets
===================================

.. contents:: mmocr.datasets.transforms
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: mmocr.datasets.transforms

Loading
---------------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   LoadImageFromFile
   LoadImageFromLMDB
   LoadOCRAnnotations
   LoadKIEAnnotations


TextDet Transforms
---------------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BoundedScaleAspectJitter
   RandomFlip
   SourceImagePad
   ShortScaleAspectJitter
   TextDetRandomCrop
   TextDetRandomCropFlip


TextRecog Transforms
---------------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   TextRecogGeneralAug
   CropHeight
   ImageContentJitter
   ReversePixels
   PyramidRescale
   PadToWidth
   RescaleToHeight


OCR Transforms
---------------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   RandomCrop
   RandomRotate
   Resize
   FixInvalidPolygon
   RemoveIgnored



Formatting
---------------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   PackTextDetInputs
   PackTextRecogInputs
   PackKIEInputs


Transform Wrapper
---------------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   ImgAugWrapper
   TorchVisionWrapper


Adapter
---------------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   MMDet2MMOCR
   MMOCR2MMDet
