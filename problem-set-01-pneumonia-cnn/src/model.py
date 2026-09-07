"""
Two architectures, so the report has something to compare.

  baseline : a small CNN trained from scratch. Answers "does the task need
             pretrained features at all?"
  transfer : MobileNetV2 pretrained on ImageNet, head trained first, then the
             top of the backbone fine-tuned at a low learning rate.
"""

from tensorflow import keras
from tensorflow.keras import layers

from dataset import augmentation_layer


def metrics():
    return [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.AUC(name="pr_auc", curve="PR"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]


def build_baseline(input_shape=(224, 224, 3), lr=1e-3):
    """Four conv blocks, BN + GAP, ~1.2M params."""
    inputs = keras.Input(shape=input_shape)
    x = augmentation_layer()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="baseline_cnn")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy", metrics=metrics())
    return model


def build_transfer(input_shape=(224, 224, 3), lr=1e-3):
    """MobileNetV2 backbone, frozen. Call unfreeze_top() for stage 2."""
    base = keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = augmentation_layer()(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)   # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)                        # keep BN in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="mobilenetv2_transfer")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy", metrics=metrics())
    return model


def get_backbone(model):
    """
    The pretrained sub-model inside `model`.

    Note the augmentation block is a Sequential, which is itself a keras.Model,
    so it has to be excluded by name or it gets picked up first.
    """
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name != "augmentation":
            return layer
    return None


def unfreeze_top(model, n_layers=40, lr=1e-5):
    """
    Stage 2: unfreeze the last n_layers of the backbone at a much smaller LR.

    BatchNorm layers stay frozen. Updating their running statistics on a small
    medical dataset with a batch size of 32 is a well-known way to wreck a
    pretrained backbone.
    """
    base = get_backbone(model)
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy", metrics=metrics())
    return model


BUILDERS = {"baseline": build_baseline, "transfer": build_transfer}
