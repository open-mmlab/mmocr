.. role:: hidden
    :class: hidden-section

mmocr.models
===================================

- :mod:`~mmocr.models.common`

  - :ref:`commombackbones`
  - :ref:`commomdictionary`
  - :ref:`commomlayers`
  - :ref:`commomlosses`
  - :ref:`commommodules`

- :mod:`~mmocr.models.textdet`

  - :ref:`detdetectors`
  - :ref:`detdatapreprocessors`
  - :ref:`detnecks`
  - :ref:`detheads`
  - :ref:`detmodulelosses`
  - :ref:`detpostprocessors`

- :mod:`~mmocr.models.textrecog`

  - :ref:`recrecognizers`
  - :ref:`recdatapreprocessors`
  - :ref:`recencoders`
  - :ref:`recdecoders`
  - :ref:`recmodulelosses`
  - :ref:`recpostprocessors`
  - :ref:`reclayers`

- :mod:`~mmocr.models.kie`

  - :ref:`kieextractors`
  - :ref:`kieheads`
  - :ref:`kiemodulelosses`
  - :ref:`kiepostprocessors`


.. module:: mmocr.models.common
models.common
---------------------------------------------
.. currentmodule:: mmocr.models.common

.. _commombackbones:

BackBones
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   UNet

.. _commomdictionary:

Dictionary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   Dictionary

.. _commomlosses:

Losses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   MaskedBalancedBCEWithLogitsLoss
   MaskedDiceLoss
   MaskedSmoothL1Loss
   MaskedSquareDiceLoss
   MaskedBCEWithLogitsLoss
   SmoothL1Loss
   CrossEntropyLoss
   MaskedBalancedBCELoss
   MaskedBCELoss

.. _commomlayers:

Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   TFEncoderLayer
   TFDecoderLayer

.. _commommodules:

Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   ScaledDotProductAttention
   MultiHeadAttention
   PositionwiseFeedForward
   PositionalEncoding


.. module:: mmocr.models.textdet
models.textdet
---------------------------------------------
.. currentmodule:: mmocr.models.textdet

.. _detdetectors:

Detectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   SingleStageTextDetector
   DBNet
   PANet
   PSENet
   TextSnake
   FCENet
   DRRG
   MMDetWrapper


.. _detdatapreprocessors:

Data Preprocessors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   TextDetDataPreprocessor


.. _detnecks:

Necks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   FPEM_FFM
   FPNF
   FPNC
   FPN_UNet


.. _detheads:

Heads
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BaseTextDetHead
   PSEHead
   PANHead
   DBHead
   FCEHead
   TextSnakeHead
   DRRGHead


.. _detmodulelosses:

Module Losses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   SegBasedModuleLoss
   PANModuleLoss
   PSEModuleLoss
   DBModuleLoss
   TextSnakeModuleLoss
   FCEModuleLoss
   DRRGModuleLoss


.. _detpostprocessors:

Postprocessors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BaseTextDetPostProcessor
   PSEPostprocessor
   PANPostprocessor
   DBPostprocessor
   DRRGPostprocessor
   FCEPostprocessor
   TextSnakePostprocessor



.. module:: mmocr.models.textrecog
models.textrecog
---------------------------------------------
.. currentmodule:: mmocr.models.textrecog

.. _recrecognizers:


Recognizers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BaseRecognizer
   EncoderDecoderRecognizer
   CRNN
   SARNet
   NRTR
   RobustScanner
   SATRN
   ABINet
   MASTER

.. _recdatapreprocessors:

Data Preprocessors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   TextRecogDataPreprocessor


.. _recbackbones:

BackBones
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   ResNet31OCR
   MiniVGG
   NRTRModalityTransform
   ShallowCNN
   ResNetABI
   ResNet
   MobileNetV2


.. _recencoders:

Encoders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   SAREncoder
   NRTREncoder
   BaseEncoder
   ChannelReductionEncoder
   SATRNEncoder
   ABIEncoder

.. _recdecoders:

Decoders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BaseDecoder
   ABILanguageDecoder
   ABIVisionDecoder
   ABIFuser
   CRNNDecoder
   ParallelSARDecoder
   SequentialSARDecoder
   ParallelSARDecoderWithBS
   NRTRDecoder
   SequenceAttentionDecoder
   PositionAttentionDecoder
   RobustScannerFuser
   MasterDecoder

.. _recmodulelosses:

Module Losses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BaseTextRecogModuleLoss
   CEModuleLoss
   CTCModuleLoss
   ABIModuleLoss

.. _recpostprocessors:

Postprocessors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BaseTextRecogPostprocessor
   AttentionPostprocessor
   CTCPostProcessor

.. _reclayers:

Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BidirectionalLSTM
   Adaptive2DPositionalEncoding
   BasicBlock
   Bottleneck
   RobustScannerFusionLayer
   DotProductAttentionLayer
   PositionAwareLayer
   SATRNEncoderLayer


.. module:: mmocr.models.kie
models.kie
---------------------------------------------
.. currentmodule:: mmocr.models.kie

.. _kieextractors:

Extractors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   SDMGR

.. _kieheads:

Heads
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   SDMGRHead

.. _kiemodulelosses:

Module Losses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   SDMGRModuleLoss

.. _kiepostprocessors:

Postprocessors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   SDMGRPostProcessor
