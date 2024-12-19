from enum import Enum


class StrategyType(str, Enum):
   PUT = "put"
   CALL = "call"
   PD_SPREAD = "put debit spread"
   PC_SPREAD = "put credit spread"
   CD_SPREAD = "call debit spread"
   CC_SPREAD = "call credit spread"