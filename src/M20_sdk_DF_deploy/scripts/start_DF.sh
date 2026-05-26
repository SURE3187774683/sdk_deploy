#!/bin/bash

# 使用 -t 参数分配伪终端，因为 sudo 通常需要 tty
ssh -t user@10.21.31.106 "sudo drmap mapping -s"