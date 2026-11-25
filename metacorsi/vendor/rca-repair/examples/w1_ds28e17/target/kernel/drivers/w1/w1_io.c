// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Subset of linux-6.18 drivers/w1/w1_io.c: the functions the DS28E17 driver calls, copied verbatim.
 * Built in user space against shim/ headers.
 */
#include <linux/w1.h>
#include "w1_io_shim.h"

static void w1_delay(unsigned long tm)
{
	udelay(tm * w1_delay_parm);
}

u8 w1_touch_bit(struct w1_master *dev, int bit)
{
	if (dev->bus_master->touch_bit)
		return dev->bus_master->touch_bit(dev->bus_master->data, bit);
	else if (bit)
		return w1_read_bit(dev);
	else {
		w1_write_bit(dev, 0);
		return 0;
	}
}

static void w1_write_bit(struct w1_master *dev, int bit)
{
	unsigned long flags = 0;

	if(w1_disable_irqs) local_irq_save(flags);

	if (bit) {
		dev->bus_master->write_bit(dev->bus_master->data, 0);
		w1_delay(6);
		dev->bus_master->write_bit(dev->bus_master->data, 1);
		w1_delay(64);
	} else {
		dev->bus_master->write_bit(dev->bus_master->data, 0);
		w1_delay(60);
		dev->bus_master->write_bit(dev->bus_master->data, 1);
		w1_delay(10);
	}

	if(w1_disable_irqs) local_irq_restore(flags);
}

static void w1_pre_write(struct w1_master *dev)
{
	if (dev->pullup_duration &&
		dev->enable_pullup && dev->bus_master->set_pullup) {
		dev->bus_master->set_pullup(dev->bus_master->data,
			dev->pullup_duration);
	}
}

static void w1_post_write(struct w1_master *dev)
{
	if (dev->pullup_duration) {
		if (dev->enable_pullup && dev->bus_master->set_pullup)
			dev->bus_master->set_pullup(dev->bus_master->data, 0);
		else
			msleep(dev->pullup_duration);
		dev->pullup_duration = 0;
	}
}

void w1_write_8(struct w1_master *dev, u8 byte)
{
	int i;

	if (dev->bus_master->write_byte) {
		w1_pre_write(dev);
		dev->bus_master->write_byte(dev->bus_master->data, byte);
	}
	else
		for (i = 0; i < 8; ++i) {
			if (i == 7)
				w1_pre_write(dev);
			w1_touch_bit(dev, (byte >> i) & 0x1);
		}
	w1_post_write(dev);
}

static u8 w1_read_bit(struct w1_master *dev)
{
	int result;
	unsigned long flags = 0;

	/* sample timing is critical here */
	local_irq_save(flags);
	dev->bus_master->write_bit(dev->bus_master->data, 0);
	w1_delay(6);
	dev->bus_master->write_bit(dev->bus_master->data, 1);
	w1_delay(9);

	result = dev->bus_master->read_bit(dev->bus_master->data);
	local_irq_restore(flags);

	w1_delay(55);

	return result & 0x1;
}

u8 w1_read_8(struct w1_master *dev)
{
	int i;
	u8 res = 0;

	if (dev->bus_master->read_byte)
		res = dev->bus_master->read_byte(dev->bus_master->data);
	else
		for (i = 0; i < 8; ++i)
			res |= (w1_touch_bit(dev,1) << i);

	return res;
}

void w1_write_block(struct w1_master *dev, const u8 *buf, int len)
{
	int i;

	if (dev->bus_master->write_block) {
		w1_pre_write(dev);
		dev->bus_master->write_block(dev->bus_master->data, buf, len);
	}
	else
		for (i = 0; i < len; ++i)
			w1_write_8(dev, buf[i]); /* calls w1_pre_write */
	w1_post_write(dev);
}

u8 w1_read_block(struct w1_master *dev, u8 *buf, int len)
{
	int i;
	u8 ret;

	if (dev->bus_master->read_block)
		ret = dev->bus_master->read_block(dev->bus_master->data, buf, len);
	else {
		for (i = 0; i < len; ++i)
			buf[i] = w1_read_8(dev);
		ret = len;
	}

	return ret;
}

int w1_reset_bus(struct w1_master *dev)
{
	int result;
	unsigned long flags = 0;

	if(w1_disable_irqs) local_irq_save(flags);

	if (dev->bus_master->reset_bus)
		result = dev->bus_master->reset_bus(dev->bus_master->data) & 0x1;
	else {
		dev->bus_master->write_bit(dev->bus_master->data, 0);
		/* minimum 480, max ? us
		 * be nice and sleep, except 18b20 spec lists 960us maximum,
		 * so until we can sleep with microsecond accuracy, spin.
		 * Feel free to come up with some other way to give up the
		 * cpu for such a short amount of time AND get it back in
		 * the maximum amount of time.
		 */
		w1_delay(500);
		dev->bus_master->write_bit(dev->bus_master->data, 1);
		w1_delay(70);

		result = dev->bus_master->read_bit(dev->bus_master->data) & 0x1;
		/* minimum 70 (above) + 430 = 500 us
		 * There aren't any timing requirements between a reset and
		 * the following transactions.  Sleeping is safe here.
		 */
		/* w1_delay(430); min required time */
		msleep(1);
	}

	if(w1_disable_irqs) local_irq_restore(flags);

	return result;
}

int w1_reset_select_slave(struct w1_slave *sl)
{
	if (w1_reset_bus(sl->master))
		return -1;

	if (sl->master->slave_count == 1)
		w1_write_8(sl->master, W1_SKIP_ROM);
	else {
		u8 match[9] = {W1_MATCH_ROM, };
		u64 rn = le64_to_cpu(*((u64*)&sl->reg_num));

		memcpy(&match[1], &rn, 8);
		w1_write_block(sl->master, match, 9);
	}
	return 0;
}

int w1_reset_resume_command(struct w1_master *dev)
{
	if (w1_reset_bus(dev))
		return -1;

	w1_write_8(dev, dev->slave_count > 1 ? W1_RESUME_CMD : W1_SKIP_ROM);
	return 0;
}

