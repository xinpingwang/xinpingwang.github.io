---
title: PostgreSQL 常用操作
date: 2020-04-21
tags:
---

**连接数据库**

```bash
psql -h host -U username [-d db_name]
```

**执行 sql 文件**

```bash
psql -h host -U username -d db_name -f filename.sql
```

**psql 交互环境下常用指令**

```
\l 查看所有可用数据库
\d 查看数据库下的所有表
\c db_name 切换到指定数据库
\password user 修改指定用户的密码
create database db_name 创建数据库
```

**配置免密码登录**

创建 `~/.pgpass` 文件，出于安全考虑，文件权限必须是 0600，文件内容格式如下：

```
hostname:port:database:username:password
```

https://stackoverflow.com/questions/6405127/how-do-i-specify-a-password-to-psql-non-interactively

https://www.postgresql.org/docs/9.3/libpq-pgpass.html

**重置 SERIAL 计数器**

```sql
SELECT setval('**_id_seq', 1, FALSE);
```

https://timmurphy.org/2009/11/19/resetting-serial-counters-in-postgresql/

**复制数据到另一个表**

```sql
-- target 表存在
insert into 目标表 select * from 原表;
-- target 表不存在
select * into 目标表 from 原表;
```

https://www.cnblogs.com/feifeicui/p/8965807.html

