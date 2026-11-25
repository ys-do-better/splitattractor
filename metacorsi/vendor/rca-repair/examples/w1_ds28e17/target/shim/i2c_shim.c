/* Minimal I2C core outside the SMBus emulation: adapter registry and __i2c_transfer (quirks check + master_xfer). */
#include <kshim.h>

#define MAX_ADAPTERS 4
struct i2c_adapter *shim_adapters[MAX_ADAPTERS];
int shim_nadapters;

int i2c_add_adapter(struct i2c_adapter *adap)
{
	if (shim_nadapters == MAX_ADAPTERS)
		return -ENOMEM;
	adap->nr = shim_nadapters;
	shim_adapters[shim_nadapters++] = adap;
	return 0;
}

void i2c_del_adapter(struct i2c_adapter *adap)
{
	for (int i = 0; i < shim_nadapters; i++)
		if (shim_adapters[i] == adap)
			shim_adapters[i] = NULL;
}

static int check_quirks(struct i2c_adapter *adap, struct i2c_msg *msgs, int num)
{
	const struct i2c_adapter_quirks *q = adap->quirks;
	for (int i = 0; i < num; i++) {
		if ((msgs[i].flags & I2C_M_RD) && q->max_read_len && msgs[i].len > q->max_read_len)
			return -EOPNOTSUPP;
		if (!(msgs[i].flags & I2C_M_RD) && q->max_write_len && msgs[i].len > q->max_write_len)
			return -EOPNOTSUPP;
	}
	return 0;
}

int __i2c_transfer(struct i2c_adapter *adap, struct i2c_msg *msgs, int num)
{
	if (!adap->algo->master_xfer)
		return -EOPNOTSUPP;
	if (!msgs || num < 1)
		return -EINVAL;
	if (adap->quirks && check_quirks(adap, msgs, num))
		return -EOPNOTSUPP;
	return adap->algo->master_xfer(adap, msgs, num);
}

int i2c_transfer(struct i2c_adapter *adap, struct i2c_msg *msgs, int num)
{
	return __i2c_transfer(adap, msgs, num);
}
