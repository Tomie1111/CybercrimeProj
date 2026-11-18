from scapy.all import RadioTap,Dot11,Dot11Beacon,Dot11Elt
from evil_twin.features.beacon import parse_beacon
from evil_twin.features.security import decode_rsn
from evil_twin.features.vendor import decode_vendor_ie


def test_beacon_and_malformed_handling():
    pkt=RadioTap()/Dot11(type=0,subtype=8,addr3="00:11:22:33:44:55")/Dot11Beacon(cap=0x11)/Dot11Elt(ID=0,info=b"lab")/Dot11Elt(ID=221,info=b"\x00\x50\xf2\x04")
    row=parse_beacon(pkt,"x",1)
    assert row["bssid"]=="00:11:22:33:44:55" and row["vendor_ie_count"]==1
    assert parse_beacon(RadioTap()/Dot11(),"x",2) is None


def test_rsn_akm_decode():
    data=(b"\x01\x00"+b"\x00\x0f\xac\x04"+b"\x01\x00"+b"\x00\x0f\xac\x04"+
          b"\x01\x00"+b"\x00\x0f\xac\x02"+b"\xc0\x00")
    x=decode_rsn(data)
    assert x["valid"] and x["akm"]==["000fac:2"] and x["pmf_capable"] and x["pmf_required"]
    assert not decode_rsn(b"\x01")["valid"]


def test_vendor_wps_decode():
    x=decode_vendor_ie(b"\x00\x50\xf2\x04\x10\x4a\x00\x01\x10")
    assert x["valid"] and x["wps_attributes"]["0x104a"]=="10"
