import tensorrt as trt
import os


class ONNX2TensorRT:

    def __init__(self, onnx_file=None, engine_file=None, precision_mode=None):
        self.onnx_file = onnx_file
        self.engine_file = engine_file
        self.precision_mode = precision_mode
        os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

    def onnx2tensor_convertor(self):
        print('Conversion of ONNX File to TRT/Engine File ......')
        # Logger to log all messages preceding a certain severity to stdout
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)  # Creating a Builder

        # Creating Network Definition.
        # EXPLICIT_BATCH flag is required in order to import models using ONNX Parser.
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        if not (self.onnx_file and self.engine_file):
            print('ONNX File or Engine File Path Missing!!')
            exit()

        else:
            # Read the Model File and Process any Errors
            success = parser.parse_from_file(self.onnx_file)
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))

            if success:
                config = builder.create_builder_config()

                # Setting Maximum Workspace Size.
                # Default Workspace is set to the Total Global Memory Size for a given Device.
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)  # 1 MB

                # Set precision mode based on argument
                if self.precision_mode == 'float16':
                    config.clear_flag(trt.BuilderFlag.TF32)
                    config.set_flag(trt.BuilderFlag.FP16, True)
                    print('Precision Set to float16')

                elif self.precision_mode == 'int8':
                    config.clear_flag(trt.BuilderFlag.TF32)
                    config.set_flag(trt.BuilderFlag.INT8, True)
                    print('Precision Set to int8')

                # Add optimization profile
                profile = builder.create_optimization_profile()
                profile.set_shape("input", (1, 3, 1824, 2752), (1, 3, 1824, 2752), (1, 3, 1824, 2752))
                config.add_optimization_profile(profile)

                print('Precision set to float32')
                serialize_engine = builder.build_serialized_network(network, config)

                if serialize_engine is None:
                    # Print any errors that occurred during the build process
                    print("Failed to build TensorRT engine. Check your network definition and settings.")
                    print(logger.log)
                else:
                    # Saving trt File for Future use
                    engine_file = self.engine_file
                    with open(engine_file, 'wb') as f:
                        f.write(serialize_engine)
                    print(f'Conversion Successful!! File Available in {self.engine_file}')

            else:
                print('Issue with Parsing the ONNX File ....')

        print('Validating Tensor-RT Model ..... ')
        try:
            with open(self.engine_file, 'rb') as f, trt.Runtime(trt.Logger(min_severity=trt.Logger.ERROR)) as runtime:
                runtime.deserialize_cuda_engine(f.read())
            print("TensorRT engine file is valid.")
        except Exception as e:
            print(f"TensorRT engine file is invalid. Error: {e}")


if __name__ == '__main__':
    o2t = ONNX2TensorRT(onnx_file='../onnx_weights/Aug21_bestmodel_run1_0.038.onnx',
                        engine_file='../tensorrt_weights/Aug21_bestmodel_run1_0.038.engine')

    o2t.onnx2tensor_convertor()
