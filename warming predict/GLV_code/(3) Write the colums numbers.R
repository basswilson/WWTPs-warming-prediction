library(readr)
library(dplyr)

#选择要处理的文件
target = "Before"
# target = "After"

# Before
path <- paste0("/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/results/", target, " Perturbed/GLV/analysis")
setwd(path)



# 定义4.csv的列名（共84个特征列）
features_colnames <- c("ENV1", "ENV2","ENV3","ENV4","ENV5","ENV6","ENV7","ENV8")

# 定义Test_log文件的列名（15列）
log_colnames <- c("Index", 
                  "Column", "Seed", "Weight_Decay", "Dropout_Probability", 
                  features_colnames, 
                  "Target", "Predicted")

# 定义Test_and_parameters文件的列名 (15列)
para_colnames <- c("Column", "Seed", "Weight_Decay", "Dropout_Probability",
                   "MSE_Test", "R2_Test",
                   paste0("RI_", 1:8), 
                   "Bias")


path1 <- paste0("Test_",target,".txt")
path2 <- paste0("Test_and_parameters_",target,".csv")
# 读取数据文件
test_log <- read_delim(path1,
                       delim = "\t", 
                       col_names = log_colnames)

test_para <- read_csv(path2,
                     col_names = para_colnames)

# 验证列数是否正确
cat("Test_log列数验证:", ncol(test_log) == 15, "\n")
cat("Test_parameters列数验证:", ncol(test_para) == 15, "\n")

# 查看数据结构
glimpse(test_log)
glimpse(test_para)



path3 <- paste0( "Test_",target,"_with_headers.csv")
path4 <- paste0("Test_and_parameters_",target,"_with_headers.csv")
# 保存带列名的文件
write_csv(test_log, 
            path3)

write_csv(test_para, 
          path4)
