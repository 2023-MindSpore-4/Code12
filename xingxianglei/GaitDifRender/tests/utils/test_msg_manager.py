class TestMSGManager:
    def test_init_manager(self, msg_mgr):
        save_path = 'output/CASIA-B/Baseline/Baseline'
        log_to_file = False
        log_iter = 0
        iteration = 0
        msg_mgr.init_manager(save_path=save_path, log_to_file=log_to_file, log_iter=log_iter, iteration=iteration)

    def test_init_logger(self, msg_mgr):
        save_path = 'output/CASIA-B/Baseline/Baseline'
        log_to_file = False
        msg_mgr.init_logger(save_path=save_path, log_to_file=False)

    def test_msg_manager(self):
        from opengait.utils.msg_manager import get_msg_mgr
        msg_mgr = get_msg_mgr()
        self.test_init_manager(msg_mgr)
        self.test_init_logger(msg_mgr)
