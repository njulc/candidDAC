from harl.common.base_logger import BaseLogger


class SIGMOIDLogger(BaseLogger):
    def get_task_name(self):
        return "sigmoid_state"
