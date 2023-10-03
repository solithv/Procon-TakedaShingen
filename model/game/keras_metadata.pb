
�aroot"_tf_keras_network*�a{"name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Functional", "config": {"name": "model_1", "trainable": true, "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 5, 11]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention_1", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 11, "value_dim": 11, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1, 2]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}, "key_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}, "value_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}}, "name": "multi_head_attention_1", "inbound_nodes": [[["input_2", 0, 0, {"value": ["input_2", 0, 0], "key": ["input_2", 0, 0]}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["multi_head_attention_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_2", "inbound_nodes": [["input_2", 0, 0, {"y": ["dropout_2", 0, 0], "name": null}]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_2", "inbound_nodes": [[["tf.__operators__.add_2", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 5, 11]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_3_input"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 11, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_1", "inbound_nodes": [[["layer_normalization_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_3", "inbound_nodes": [[["sequential_1", 1, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_3", "inbound_nodes": [["layer_normalization_2", 0, 0, {"y": ["dropout_3", 0, 0], "name": null}]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_3", "inbound_nodes": [[["tf.__operators__.add_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["layer_normalization_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 17, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_5", 0, 0]]}, "shared_object_id": 26, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 5, 5, 11]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 5, 5, 11]}, "float32", "input_2"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 5, 5, 11]}, "float32", "input_2"]}, "keras_version": "2.13.1", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "trainable": true, "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 5, 11]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention_1", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 11, "value_dim": 11, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1, 2]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}, "key_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}, "value_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}}, "name": "multi_head_attention_1", "inbound_nodes": [[["input_2", 0, 0, {"value": ["input_2", 0, 0], "key": ["input_2", 0, 0]}]]], "shared_object_id": 3}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["multi_head_attention_1", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_2", "inbound_nodes": [["input_2", 0, 0, {"y": ["dropout_2", 0, 0], "name": null}]], "shared_object_id": 5}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_2", "inbound_nodes": [[["tf.__operators__.add_2", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 5, 11]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_3_input"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 11, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_1", "inbound_nodes": [[["layer_normalization_2", 0, 0, {}]]], "shared_object_id": 16}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_3", "inbound_nodes": [[["sequential_1", 1, 0, {}]]], "shared_object_id": 17}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_3", "inbound_nodes": [["layer_normalization_2", 0, 0, {"y": ["dropout_3", 0, 0], "name": null}]], "shared_object_id": 18}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 20}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_3", "inbound_nodes": [[["tf.__operators__.add_3", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["layer_normalization_3", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 17, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["flatten_1", 0, 0, {}]]], "shared_object_id": 25}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_5", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 28}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Custom>Adam", "config": {"name": "Adam", "weight_decay": null, "clipnorm": null, "global_clipnorm": null, "clipvalue": null, "use_ema": false, "ema_momentum": 0.99, "ema_overwrite_frequency": null, "jit_compile": true, "is_legacy_optimizer": false, "learning_rate": 0.0010000000474974513, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 5, 11]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 5, 11]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}2
�
root.layer_with_weights-0"_tf_keras_layer*�	{"name": "multi_head_attention_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention_1", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 11, "value_dim": 11, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1, 2]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}, "key_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}, "value_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}}, "inbound_nodes": [[["input_2", 0, 0, {"value": ["input_2", 0, 0], "key": ["input_2", 0, 0]}]]], "shared_object_id": 3, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}}2
�root.layer-2"_tf_keras_layer*�{"name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "inbound_nodes": [[["multi_head_attention_1", 0, 0, {}]]], "shared_object_id": 4, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}}2
�root.layer-3"_tf_keras_layer*�{"name": "tf.__operators__.add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "preserve_input_structure_in_config": true, "autocast": false, "class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "inbound_nodes": [["input_2", 0, 0, {"y": ["dropout_2", 0, 0], "name": null}]], "shared_object_id": 5, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "layer_normalization_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["tf.__operators__.add_2", 0, 0, {}]]], "shared_object_id": 8, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}}2
�root.layer_with_weights-2"_tf_keras_sequential*�{"name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 5, 11]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_3_input"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 11, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "inbound_nodes": [[["layer_normalization_2", 0, 0, {}]]], "shared_object_id": 16, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 5, 5, 11]}, "float32", "dense_3_input"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 5, 5, 11]}, "float32", "dense_3_input"]}, "keras_version": "2.13.1", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 5, 11]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_3_input"}, "shared_object_id": 9}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 11, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15}]}}}2
�root.layer-6"_tf_keras_layer*�{"name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "inbound_nodes": [[["sequential_1", 1, 0, {}]]], "shared_object_id": 17, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}}2
�root.layer-7"_tf_keras_layer*�{"name": "tf.__operators__.add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "preserve_input_structure_in_config": true, "autocast": false, "class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "inbound_nodes": [["layer_normalization_2", 0, 0, {"y": ["dropout_3", 0, 0], "name": null}]], "shared_object_id": 18, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}}2
�	root.layer_with_weights-3"_tf_keras_layer*�{"name": "layer_normalization_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 20}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["tf.__operators__.add_3", 0, 0, {}]]], "shared_object_id": 21, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}}2
�
root.layer-9"_tf_keras_layer*�{"name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["layer_normalization_3", 0, 0, {}]]], "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}}2
�root.layer_with_weights-4"_tf_keras_layer*�{"name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 17, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_1", 0, 0, {}]]], "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 275}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 275]}}2
�&root.layer_with_weights-0._query_dense"_tf_keras_layer*�{"name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "EinsumDense", "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, null, 4, 11], "equation": "abcd,def->abcef", "activation": "linear", "bias_axes": "ef", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 33, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}}2
�$root.layer_with_weights-0._key_dense"_tf_keras_layer*�{"name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "EinsumDense", "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, null, 4, 11], "equation": "abcd,def->abcef", "activation": "linear", "bias_axes": "ef", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}}2
�&root.layer_with_weights-0._value_dense"_tf_keras_layer*�{"name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "EinsumDense", "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, null, 4, 11], "equation": "abcd,def->abcef", "activation": "linear", "bias_axes": "ef", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 39, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}}2
�"root.layer_with_weights-0._softmax"_tf_keras_layer*�{"name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [4, 5]}}, "shared_object_id": 40, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 5, 5, 5, 5]}}2
�(root.layer_with_weights-0._dropout_layer"_tf_keras_layer*�{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "shared_object_id": 41, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 5, 5, 5, 5]}}2
� 'root.layer_with_weights-0._output_dense"_tf_keras_layer*�{"name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "EinsumDense", "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, null, 11], "equation": "abcde,def->abcf", "activation": "linear", "bias_axes": "f", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 42}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 43}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 44, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 4, 11]}}2
�2.root.layer_with_weights-2.layer_with_weights-0"_tf_keras_layer*�{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 11}}, "shared_object_id": 45}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 11]}}2
�3.root.layer_with_weights-2.layer_with_weights-1"_tf_keras_layer*�{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 11, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 32]}}2
��root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 47}2
��root.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 28}2