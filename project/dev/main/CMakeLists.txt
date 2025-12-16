idf_component_register(SRCS "i2s_main.c"
                    PRIV_REQUIRES esp_driver_i2s esp_driver_gpio i2s_examples_common esp_timer esp_partition partition_table spi_flash
                    INCLUDE_DIRS ".")
