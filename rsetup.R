install.packages(c("docopt", "BiocManager"), repos='https://stat.ethz.ch/CRAN/')
install.packages("remotes", dependencies=TRUE, repos='https://stat.ethz.ch/CRAN/')
library(remotes)
# Versions specifically for R 4.3.2
install_version("MASS", "7.3.60") # will ask for mirrors: select 65
install_version("Matrix", "1.6.5")
BiocManager::install(c("igraph", "RBGL", "ggm", "Rgraphviz"), update=TRUE, ask=FALSE)
install.packages(c("momentchi2", "pcalg", "kpcalg"), dependencies=TRUE, repos='https://stat.ethz.ch/CRAN/')
install_github("Diviyan-Kalainathan/RCIT")
