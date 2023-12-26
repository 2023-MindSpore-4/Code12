# import test_case.mindnlp_test as mindnlp_test

import mindspore as ms
ms.set_context(mode = ms.PYNATIVE_MODE)
ms.set_context(device_target = "GPU")
ms.ms_memory_recycle()

def main():
    import test_case.data_loader as data_loader_test
    data_loader_test.test_get_data()
    # data_loader_test.file_type_test()
    # data_loader_test.test_get_data()
    # data_loader_test.test_path()
    # model_inference_test.show_pertrain_info()
    # model_inference_test.show_torch_pertrain_info()

    # import test_case.model_inference as model_inference_test
    # model_inference_test.train_test()
    # model_inference_test.inference()
    # model_inference_test.show_pertrain_info()
    # mindnlp_test.test_Bert()

    # import test_case.mindspore_hub_test as mshub_test
    # mshub_test.test_install_bert()

    # import test_case.mindformer_test as mft
    # mft.test_load_bert()
    pass

if __name__ == '__main__':
    main()