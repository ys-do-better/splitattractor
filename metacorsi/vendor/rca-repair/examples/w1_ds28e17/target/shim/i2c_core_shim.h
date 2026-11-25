/* File-scope pieces of drivers/i2c/i2c-core-smbus.c and i2c-core.h outside the copied functions. */
#ifndef I2C_CORE_SHIM_H
#define I2C_CORE_SHIM_H
#define POLY (0x1070U << 3)
#define I2C_LOCK_SEGMENT 2
static inline int __i2c_lock_bus_helper(struct i2c_adapter *adap) { (void)adap; return 0; }
static inline void i2c_unlock_bus(struct i2c_adapter *adap, unsigned int flags) { (void)adap; (void)flags; }
static inline int __i2c_check_suspended(struct i2c_adapter *adap) { (void)adap; return 0; }
static inline bool i2c_in_atomic_xfer_mode(void) { return false; }
static unsigned long jiffies;
#define time_after(a, b) ((long)((b) - (a)) < 0)
#define trace_smbus_write(...) do { } while (0)
#define trace_smbus_read(...) do { } while (0)
#define trace_smbus_reply(...) do { } while (0)
#define trace_smbus_result(...) do { } while (0)
#endif
