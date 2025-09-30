library(readr)
library(dplyr)

# 服务器上运行时请将
# /Users/basswilson/PycharmProjects/pythonProject/WWTPs_Tem_DIS_(Final)
# 改为
# /home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)


rm(list=ls())
# target = "Before"
target = "After"

# Before
path <- paste0("/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/results/", target, " Perturbed/alpha/analysis")
setwd(path)

# 定义4.csv的列名（共84个特征列）
features_colnames <- c("Latitude", "Longitude", 
                       "Country.RegionAustralia", "Country.RegionAustria", 
                       "Country.RegionBrazil", "Country.RegionCada", 
                       "Country.RegionChi", "Country.RegionChile", 
                       "Country.RegionColombia", "Country.RegionDenmark", 
                       "Country.RegionItaly", "Country.RegionMexico", 
                       "Country.RegionSingapore", "Country.RegionSouth Africa", 
                       "Country.RegionSweden", "Country.RegionSwitzerland", 
                       "Country.RegionTaiwan (ROC)", "Country.RegionUnited Kindom", 
                       "Country.RegionUnited States", "Country.RegionUruguay", 
                       "ContinentAfrica", "ContinentAsia", 
                       "ContinentAustralasia", "ContinentEurope", 
                       "ContinentNorth America", "ContinentSouth America", 
                       "Climate.typeAf", "Climate.typeAm", "Climate.typeAw", 
                       "Climate.typeBWh", "Climate.typeCfa", "Climate.typeCfb", 
                       "Climate.typeCsa", "Climate.typeCsb", "Climate.typeCwa", 
                       "Climate.typeCwb", "Climate.typeCwc", "Climate.typeDfa", 
                       "Climate.typeDfb", "Climate.typeDsc", "Climate.typeDwa", 
                       "Climate.typeET", "Annual.average", 
                       "Annual.mean.of.daily.maximum", "Annual.mean.of.daily.minimum", 
                       "Sampling.month.average", "Precipitation.Annual", 
                       "Precipitation.Sampling.month", "GDP.per.capita..dollars.", 
                       "City.population", "Actual.Inf.rate..m3.d.", 
                       "Volume.of.aeration.tanks..m3.", "HRT.Plant", 
                       "HRT.Aeration.tank", "SRT..d.", "General.type", 
                       "General.typeA2O", "General.typeAO", 
                       "General.typecomplete mix", "General.typecontact stabilization", 
                       "General.typeextended aeration", "General.typeoxidation ditch", 
                       "General.typeplug flow", "General.typeSBR", 
                       "BOD.Inf", "BOD.Eff", "COD.Inf", "COD.Eff", 
                       "F.M", "NH4.N.Inf", "NH4.N.Eff", "TN.Inf", 
                       "TN.Eff", "TP.Inf", "TP.Eff", 
                       "Industry.Contained.in.Inf", 
                       "Industry.Contained.in.InfNo", 
                       "Industry.Contained.in.InfUnknown", 
                       "Industry.Contained.in.InfYes", 
                       "Industrial.Percentage", "MLSS", "DO", "pH", 
                       "Mixed.liquid.temperature..oC.")

# 定义Test_log文件的列名（总列数91）
log_colnames <- c("Index", 
                  "Column", "Seed", "Weight_Decay", "Dropout_Probability", 
                  features_colnames, 
                  "Target", "Predicted")

# 定义Test_and_parameters文件的列名（总列数91）
para_colnames <- c("Column", "Seed", "Weight_Decay", "Dropout_Probability",
                   "MSE_Test", "R2_Test",
                   paste0("RI_", 1:84), 
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
cat("Test_log列数验证:", ncol(test_log) == 91, "\n")
cat("Test_parameters列数验证:", ncol(test_para) == 91, "\n")

# 查看数据结构
glimpse(test_log)
glimpse(test_para)



# 查看RI
b <- as.data.frame(colMeans(test_para[,7:90]))
rownames(b) <- features_colnames

path3 <- paste0( "Test_",target,"_with_headers.csv")
path4 <- paste0("Test_and_parameters_",target,"_with_headers.csv")
# 保存带列名的文件
write_csv(test_log, 
          path3)

write_csv(test_para, 
          path4)
