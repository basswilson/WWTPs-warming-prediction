rm(list=ls())

# TI计算
library(tidyr)
library(ggplot2)
library(dplyr)
library(stringr)
library(data.table)

# 服务器上运行时请将 
# /Users/basswilson/PycharmProjects/pythonProject/WWTPs_Tem_DIS_(Final)
# 改为
# /home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)


################## alpha_CNN_TI ######################
# target = "GLV"
target = "alpha"

path1 = paste0("/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/results/After Perturbed/",target,"/analysis/Test_After_with_headers.csv")
path2 = paste0("/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/results/Before Perturbed/",target,"/analysis/Test_Before_with_headers.csv")

data_after <- fread(file = path1,header = "auto")
data_before <- fread(file = path2,header = "auto")



draw_data <- as.data.frame(cbind(data_after$Seed,data_after$Index,data_after$Column,data_after$Predicted,data_before$Predicted))
colnames(draw_data) = c("seed","Sample","Bacteria","after Predicted","before Predicted")


# 选取seed=0的数据
draw_data <- draw_data[draw_data$seed == 0,]


index <- read.csv("/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/污水厂数据集/index.csv")
colnames(index) <- "Sample"

library(dplyr)
filtered_df <- draw_data %>%
  filter(Bacteria %in% index$Sample)

filtered_df$TI_CNN <- (filtered_df$`after Predicted` - filtered_df$`before Predicted`) / 
  (filtered_df$`after Predicted` + filtered_df$`before Predicted`)

# 作箱线图，根据箱体是否大于0划分颜色

# 计算每个分组的中位数
median_alpha_CNN <- aggregate(TI_CNN ~ Bacteria, data = filtered_df, FUN = median)


# 对TI_CNN的绝对值排序，然后选择前100个菌
sorted_data <- median_alpha_CNN[order(-abs(median_alpha_CNN$TI_CNN)),]
a <- sorted_data[1:100,]
target <- filtered_df %>%
  filter(Bacteria %in% a$Bacteria)


# 根据中位数是否大于 0 来创建颜色向量
colors <- ifelse(median_alpha_CNN$TI_CNN >= 0, "red", "blue")
names(colors) <- median_alpha_CNN$Bacteria

# 绘制箱线图
p2 <- ggplot(target, aes(x = factor(Bacteria), y = TI_CNN)) +
  geom_boxplot(outlier.shape = NA, aes(fill = factor(Bacteria))) +
  scale_fill_manual(values = colors) +
  labs(x = "Bacteria",
       y = "TI value") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        legend.position = "none")

path3 = paste0("/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/results/",target,"_CNN_TI.png")
ggsave(plot = p2, path3, height = 7,width = 10)

####绘制539个菌的EI频率直方图

# 绘制直方图并添加标注
p3 <- ggplot(median_alpha_CNN, aes(x = TI_CNN)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black", aes(y = after_stat(count))) +
  geom_text(stat = "bin", bins = 30, aes(label = after_stat(count), y = after_stat(count)), 
            vjust = -0.5, size = 3) +
  labs(x = "TI_CNN", y = "Frequency", title = "Histogram of TI_CNN values")+
  theme_bw()


path4 = paste0("/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/results/",target,"_CNN_TI_histogram.png")
ggsave(plot = p3, path4, height = 4,width = 6)



#绘制频率饼图
# 计算大于 0、和小于 0 的个数
greater_than_zero <- sum(median_alpha_CNN$TI_CNN > 0)
less_than_zero <- sum(median_alpha_CNN$TI_CNN < 0)

# 计算占比
total <- nrow(median_alpha_CNN)
greater_percentage <- greater_than_zero / total * 100
less_percentage <- less_than_zero / total * 100

# 创建包含占比的数据框
pie_data <- data.frame(
  Category = c("Positive EI", "Negetive EI"),
  Percentage = c(greater_percentage,less_percentage)
)

# 自定义颜色
custom_colors <- c("lightblue", "#fc8d62", "#8da0cb")

# 绘制美观的饼图
p4 <- ggplot(pie_data, aes(x = "", y = Percentage, fill = Category)) +
  geom_bar(stat = "identity", width = 1, color = "white", size = 0.5) +
  coord_polar("y", start = 0) +
  labs(title = "TI frequency", fill = "") +
  scale_fill_manual(values = custom_colors) +
  theme_void() +
  geom_text(aes(label = paste0(round(Percentage, 2), "%")), 
            position = position_stack(vjust = 0.5), 
            size = 4, 
            fontface = "bold", 
            color = "white") 

path5 = paste0("/home/hongchang/Documents/WS-Pytorch/WWTPs_Tem_DIS_(Final)/results/",target,"_CNN_TI_pie.png")
ggsave(plot = p4, path5, height = 4,width = 6)
