monitor-to-full-frame.py
========================

I was recently at the doctors office for a family member who needed an EEG. I asked if we could get the EEG data afterward and they told me that they don't do that.

So instead I recorded the screen with my phone.

This script tries to find the screen and zoom in on it.

It also currently strips out the audio track and adds an audio levels meter instead.
That seems more useful for matching up EEG data with ambient noise to me.

The script isn't perfect. It just finds the biggest bright object in the room and assumes it's the monitor. If there's a head blocking
part of the monitor or something it won't detect the monitor and just falls back to showing the unprocessed frame.

This was a one day hack so don't expect much :-)
