
### Подключение EV3 к Raspberry Pi

Подключите через miniUSB кабель от блока EV3 к плате Rapsberry Pi:

Войдите на Raspberry Pi через WiFi по SSH и проверьте подключённые устройства:
```
ssh <username>@<ip>
ip a
```
Обычно, подключённое устройство должно называться eth1.

Назначьте IP этому устройству:
```
sudo ifconfig eth1 192.168.3.1 netmask 255.255.255.0 up
```

Войдите на EV3 через WiFI по SSH и посмотрите подключённые устройства:

```
ssh robot@<ip>
ip a
```
Обычно, подключённое устройство должно называться usb1.

Назначьте IP этому устройству:
```
sudo ifconfig usb1 192.168.3.2 netmask 255.255.255.0 up
```

Проверить подключение из Raspberry Pi на EV3:
```
ping 192.168.3.2
```

Подключитесь из Raspberry Pi на EV3 через SSH:
```
ssh robot@192.168.3.2
```
Пароль: maker

Можно создать проект с программами, копировать его на Raspberry Pi, а затем переносить программы на EV3:

Скопировать программу на EV3 из Raspberry:
```
scp program.py robot@192.168.3.2:/home/robot/
```

Запустить программу на EV3 через Raspberry:
```
ssh robot@192.168.3.2 "python3 /home/stem/main.py"
```

### Подключение к Raspberry Pi через LAN

Установите IP для сетевой карты Raspberry Pi:
```
sudo ifconfig eth0 192.168.4.1 netmask 255.255.255.0 up
```

Установите DHCP-сервер:
```
sudo apt install isc-dhcp-server -y
```

Укажите нужный интерфейс для DHCP в настройках:
```
sudo nano /etc/default/isc-dhcp-server
```
Измените содержимое:
```
INTERFACESv4="eth0"
```

Создаёте конфигурацию DHCP:
```
sudo nano /etc/dhcp/dhcpd.conf
```
Измените содержимое:
```
subnet 192.168.4.0 netmask 255.255.255.0 {
    range 192.168.4.100 192.168.4.200;
    option routers 192.168.4.1;
    option domain-name-servers 8.8.8.8, 8.8.4.4;
}
```

Перезапустите DHCP-сервер:
```
sudo systemctl restart isc-dhcp-server
```

Переподключитесь по LAN с компьютера