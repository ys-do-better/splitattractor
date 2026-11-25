/* File-scope pieces of drivers/w1/w1_io.c outside the copied functions (module parameters, forward declarations). */
#ifndef W1_IO_SHIM_H
#define W1_IO_SHIM_H
static int w1_delay_parm = 1;
static int w1_disable_irqs = 0;
static void w1_write_bit(struct w1_master *dev, int bit);
static u8 w1_read_bit(struct w1_master *dev);
#endif
