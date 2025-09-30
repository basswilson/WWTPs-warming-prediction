rm(list=ls())

# TI计算
library(tidyr)
library(ggplot2)
library(dplyr)
library(stringr)

################## GLV_TI ######################
GLV_Before <- read.csv("/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/GLV 数据集/人工数据集.csv")
GLV_After <- read.csv("/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/GLV 数据集/人工数据集_perturb.csv")

#转化为长表
GLV_Before_long <- GLV_Before %>%
  pivot_longer(
    cols = -X, 
    names_to = "bacteria", 
    values_to = "Before"
  )

GLV_After_long <- GLV_After %>%
  pivot_longer(
    cols = -X, 
    names_to = "bacteria", 
    values_to = "After"
  )

#合并表格
draw_glv_long <- cbind(GLV_Before_long,GLV_After_long)
colnames(draw_glv_long) <- c("Sample","Bacteria","Before","Sample","Bacteria2","After")
draw_glv_long <- draw_glv_long[,-4:-5]
draw_glv_long <- draw_glv_long[order(draw_glv_long$Bacteria), ]

#如果before为0，那么after也为0
draw_glv_long$After <- ifelse(draw_glv_long$Before == 0, 0, draw_glv_long$After)

# TI计算
draw_glv_long$TI_glv <- (draw_glv_long$After - draw_glv_long$Before) / 
  (draw_glv_long$After + draw_glv_long$Before) 

#定义函数，方便之后的NaN值替换
replace_nan <- function(x) {
  x[is.nan(x)] <- 0
  return(x)
}
# 使用 lapply 遍历draw_glv的每一列并替换 NaN
draw_glv_long <- as.data.frame(lapply(draw_glv_long, replace_nan))

draw_glv_long$Bacteria <- str_remove(draw_glv_long$Bacteria, "Bacteria_")
draw_glv_long$Sample <- str_remove(draw_glv_long$Sample, "Sample_")

# 计算每个分组的中位数
median_GLV <- aggregate(TI_glv ~ Bacteria, data = draw_glv_long, FUN = median)
median_GLV$Bacteria <- as.numeric(median_GLV$Bacteria)
# 按照 Bacteria 列进行升序排序
median_GLV <- median_GLV[order(median_GLV$Bacteria), ]


# 根据中位数是否大于 0 来创建颜色向量
colors <- ifelse(median_GLV$TI_glv > 0, "red", "blue")
names(colors) <- median_GLV$Bacteria

# 绘制箱线图
# 手动指定因子水平顺序
draw_glv_long$Bacteria <- factor(draw_glv_long$Bacteria, levels = as.character(1:100))

# 绘制图形
p1 <- ggplot(draw_glv_long, aes(x = Bacteria, y = TI_glv)) +
  geom_boxplot(outlier.shape = NA, aes(fill = Bacteria)) +
  scale_fill_manual(values = colors) +
  labs(x = "Bacteria",
       y = "TI Value") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        legend.position = "none")

ggsave(plot = p1, "/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/results/GLV_TI.png",height = 5,width = 10)


################## GLV_CNN_TI ######################
target = "GLV"
# target = "alpha"

path1 = paste0("/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/results/After Perturbed/",target,"/analysis/Test_After_with_headers.csv")
path2 = paste0("/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/results/Before Perturbed/",target,"/analysis/Test_Before_with_headers.csv")

data_after <- read.csv(path1)
data_before <- read.csv(path2)

draw_data <- as.data.frame(cbind(data_after$Index,data_after$Column,data_after$Predicted,data_before$Predicted))
colnames(draw_data) = c("Sample","Bacteria","after Predicted","before Predicted")

draw_data$TI_CNN <- (draw_data$`after Predicted` - draw_data$`before Predicted`) / 
  (draw_data$`after Predicted` + draw_data$`before Predicted`)

# 作箱线图，根据箱体是否大于0划分颜色

# 计算每个分组的中位数
median_GLV_CNN <- aggregate(TI_CNN ~ Bacteria, data = draw_data, FUN = median)

# 根据中位数是否大于 0 来创建颜色向量
colors <- ifelse(median_GLV_CNN$TI_CNN >= 0, "red", "blue")
names(colors) <- median_GLV_CNN$Bacteria

# 绘制箱线图
p2 <- ggplot(draw_data, aes(x = factor(Bacteria), y = TI_CNN)) +
  geom_boxplot(outlier.shape = NA, aes(fill = factor(Bacteria))) +
  scale_fill_manual(values = colors) +
  labs(x = "Bacteria",
       y = "TI value") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        legend.position = "none")

path3 = paste0("/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/results/",target,"_CNN_TI.png")
ggsave(plot = p2, path3, height = 7,width = 10)



######将p1和p2做到一个图里面#########
draw_data$source <- "TI_CNN"
# 为 draw_glv_long 添加一个新列来标识数据来源
draw_glv_long$source <- "TI_GLV"
colnames(draw_glv_long) <- c("Sample","Bacteria","after Predicted","before Predicted","TI","source")
colnames(draw_data) <- c("Sample","Bacteria","after Predicted","before Predicted","TI","source")
combined_data <- rbind(draw_glv_long,draw_data)  #合并表格
combined_data$source <- factor(combined_data$source, levels = c("TI_GLV", "TI_CNN"))

# 绘制合并后的箱线图
p3 <- ggplot(combined_data, aes(x = factor(Bacteria), y = TI, fill = source)) +
  geom_boxplot(outlier.shape = NA, position = position_dodge(0.9)) +
  scale_fill_manual(values = c("#1E90FF","#FFD700")) +
  labs(x = "Bacteria",
       y = "TI value") +
  theme_bw() +
  # 去除网格线
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1),
    legend.title = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    # 设置图例位置在右上角
    legend.position = c(1, 1), 
    # 设置图例的对齐方式为右上角
    legend.justification = c(1.1, 1.1) 
  )

# 显示图形
print(p3)

# 显示图形
print(p3)

ggsave(plot = p3, "/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/results/All_TI.png",height = 5,width = 10)


######计算正确率

##计算扰动后CNN是否能正确预测GLV的符号变化
result_table <- data.frame(GLV = sign(median_GLV$TI), GLV_CNN = sign(median_GLV_CNN$TI))

##去除0值物种的影响，如果GLV的TI为0，那么GLV_CNN也为0
result_table$GLV_CNN <- ifelse(result_table$GLV == 0, 0, result_table$GLV_CNN)
correct <- result_table$GLV_CNN == result_table$GLV

# 计算正确率
accuracy_rate <- sum(correct) / length(correct)
print(accuracy_rate)
