#------------------------------------------------------------------------------#
# Install the required packages
#------------------------------------------------------------------------------#

required_packages <- c("ggplot2", "gridExtra", "dplyr", "reshape2", "ggpubr")

# Check and install missing packages
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(paste("Installing", pkg, "..."))
    install.packages(pkg)
  } else {
    message(paste(pkg, "is already installed."))
  }
}
