from flask import current_app as app, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from .models import db, File, User, Log
from .module.m1.code_review import CodeReviewer
from .module.m1.parse_data import parse_log_data
from .module.m2.webConnect import CodeEvaluation, host
from .module.m3.monitor import Monitor
from .module.m4.webConnect import getInfo
import os
from datetime import timedelta


@app.route('/')
def index():
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        user = User.query.filter_by(uname=username).first()
        if not user:
            user = User(uname=username)
            db.session.add(user)
            db.session.commit()
        session['username'] = username
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


# UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {"py", "java", "c", "cpp", "js", "ts", "go", "php", "rb", "swift", "kt", "rs", "scala", "html",
                     "css", "scss", "less", "sass", "json", "yaml", "xml", "md", "txt"}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/code_review', methods=['GET', 'POST'])
def code_review():
    if 'username' in session:
        if request.method == 'POST':
            file = request.files.get('file')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                user = User.query.filter_by(uname=session['username']).first()
                new_file = File(fname=filename, uid=user.uid)
                db.session.add(new_file)
                db.session.commit()
                file_path = new_file.save_to_disk(file.read())

                # 处理代码审查逻辑
                model = CodeReviewer()
                res = model.code_review_task("file", file_path)
                errors = parse_log_data("\n\n".join(model.report_classification("error")))
                warnings = parse_log_data("\n\n".join(model.report_classification("warning")))
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()

                flash('文件上传成功，代码审查完毕', 'success')

                # 创建日志记录
                log_content = f"用户 '{session['username']}' 提交了文件 '{filename}' 的代码审查结果：\n\n"
                log_content += "### Error 输出 ###\n"
                for error in errors:
                    log_content += f"- {error}\n"
                log_content += "\n### Warning 输出 ###\n"
                for warning in warnings:
                    log_content += f"- {warning}\n"

                new_log = Log(logcontent=log_content, logtype="code_review")
                db.session.add(new_log)
                db.session.commit()

                session['review_completed'] = True
                session['errors'] = errors
                session['warnings'] = warnings
                session['source_code'] = source_code
                session['last_uploaded_file'] = filename

                return redirect(url_for('code_review'))

        return render_template('code_review.html')
    return redirect(url_for('login'))


@app.route('/code_review_result')
def code_review_result():
    if 'username' in session and session.get('review_completed'):
        errors = session.get('errors', [])
        warnings = session.get('warnings', [])
        source_code = session.get('source_code', '')

        return render_template('code_review_result.html', errors=errors, warnings=warnings, source_code=source_code)
    return redirect(url_for('code_review'))


@app.route('/code_evaluation', methods=['GET', 'POST'])
def code_evaluation():
    if 'username' in session:
        if request.method == 'POST':
            file = request.files.get('file')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                user = User.query.filter_by(uname=session['username']).first()
                new_file = File(fname=filename, uid=user.uid)
                db.session.add(new_file)
                db.session.commit()
                file_path = new_file.save_to_disk(file.read())

                # 处理代码质量评估逻辑
                log = CodeEvaluation(os.path.basename(file_path))

                # 创建日志记录
                log_content = f"用户 '{session['username']}' 提交了文件 '{filename}' 的代码质量评估日志：\n\n"
                log_content += log

                new_log = Log(logcontent=log_content, logtype="code_evaluation")
                db.session.add(new_log)
                db.session.commit()

                session['evaluation_completed'] = True
                session['last_uploaded_file'] = filename  # 保留文件名

                flash('文件上传成功，代码质量评估完毕', 'success')
                return redirect(url_for('code_evaluation'))

        return render_template('code_evaluation.html')
    return redirect(url_for('login'))

@app.route('/code_evaluation_result')
def code_evaluation_result():
    return redirect(f"http://{host}:9000/dashboard?id=test001_")


@app.route('/fault_prediction', methods=['GET', 'POST'])
def fault_prediction():
    if 'username' in session:
        if request.method == 'POST':
            time = Monitor.get_elapsed_minute()
            if time == session.get('last_time', -2):
                flash('无需重复更新', 'info')
                return redirect(url_for('fault_prediction'))
            cpu_res_x, cpu_res_y, cpu_err = Monitor.get_cpu()
            dram_errors = Monitor.get_dram()
            # print('cpu_res_x:', cpu_res_x)

            cpu_res_x = [int(x) for x in cpu_res_x]
            cpu_res_y = [float(y) for y in cpu_res_y]
            cpu_err = bool(cpu_err)
            dram_errors = [str(error) for error in dram_errors]

            # 创建日志记录
            log_content = f"用户 '{session['username']}' 更新了故障预测状态：\n"
            log_content += f"CPU 利用率情况：{cpu_res_y}\n"
            log_content += f"CPU 是否存在异常：{'是' if cpu_err else '否'}\n"
            log_content += f"预测 DRAM 故障：{'，'.join(dram_errors) if dram_errors else '无异常'}\n"

            new_log = Log(logcontent=log_content, logtype="fault_prediction")
            db.session.add(new_log)
            db.session.commit()

            session['last_time'] = time
            session['cpu_res_x'] = cpu_res_x
            session['cpu_res_y'] = cpu_res_y
            session['cpu_err'] = cpu_err
            # print(cpu_res_x)
            # print(cpu_res_y)
            session['dram_errors'] = dram_errors

            flash('故障预测已更新', 'success')
            return redirect(url_for('fault_prediction'))

        return render_template('fault_prediction.html',
                               cpu_res_x=session.get('cpu_res_x', []),
                               cpu_res_y=session.get('cpu_res_y', []),
                               cpu_err=session.get('cpu_err', False),
                               dram_errors=session.get('dram_errors', []))

    return redirect(url_for('login'))


@app.route('/fault_detection', methods=['GET', 'POST'])
def fault_detection():
    if 'username' in session:
        if request.method == 'POST':
            master_ip = host
            infoDict = getInfo()
            cluster_info = infoDict.get('cluster_status', {})
            # print(cluster_info)
            uptime_seconds = infoDict.get('uptime', 0)
            uptime_timedelta = timedelta(seconds=uptime_seconds)

            # 记录日志
            log_content = f"用户 '{session['username']}' 访问了故障检测模块\n"
            log_content += f"当前 Master 主机信息：主机 IP - {master_ip}，运行时长 - {str(uptime_timedelta)}\n"
            log_content += "集群信息：\n"
            for ip, status in cluster_info.items():
                log_content += f"- 主机 IP: {ip}，状态: {status}\n"
            log_content += f"内存总量：{infoDict.get('mem_total_kb', 0) // 1024} MB\n"
            log_content += f"内存可用：{infoDict.get('mem_free_kb', 0) // 1024} MB\n"
            log_content += f"内存可用量：{infoDict.get('mem_available_kb', 0) // 1024} MB\n"

            new_log = Log(logcontent=log_content, logtype="fault_detection")
            db.session.add(new_log)
            db.session.commit()

            session['infoDict'] = infoDict
            session['master_ip'] = master_ip
            session['cluster_info'] = cluster_info
            session['startup_time'] = str(uptime_timedelta)
            session['mem_total'] = infoDict.get('mem_total_kb', 0) // 1024
            session['mem_free'] = infoDict.get('mem_free_kb', 0) // 1024
            session['mem_available'] = infoDict.get('mem_available_kb', 0) // 1024

            flash('故障检测信息已更新', 'success')
            return redirect(url_for('fault_detection'))

        return render_template('fault_detection.html',
                               master_ip=session.get('master_ip', ''),
                               cluster_info=session.get('cluster_info', {}),
                               startup_time=session.get('startup_time', ''),
                               mem_total=session.get('mem_total', 0),
                               mem_free=session.get('mem_free', 0),
                               mem_available=session.get('mem_available', 0))

    return redirect(url_for('login'))



@app.route('/cluster_logs')
def cluster_logs():
    if 'username' in session:
        infoDict = session.get('infoDict', {})
        # print(infoDict['buddy_system_info'])
        sorted_buddy_system_info = dict(sorted(infoDict['buddy_system_info'].items(), key=lambda item: int(item[0])))
        infoDict['buddy_system_info'] = sorted_buddy_system_info
        return render_template('cluster_logs.html', infoDict=infoDict)

    return redirect(url_for('login'))



@app.route('/log_query', methods=['GET', 'POST'])
def log_query():
    if 'username' in session:
        # 获取查询参数
        keyword = request.args.get('keyword', '')
        logtypes = request.args.getlist('logtype')
        order = request.args.get('order', 'desc')

        # 构建查询
        query = Log.query
        if keyword:
            query = query.filter(Log.logcontent.contains(keyword))
        if logtypes:
            query = query.filter(Log.logtype.in_(logtypes))
        if order == 'asc':
            query = query.order_by(Log.logtime.asc())
        else:
            query = query.order_by(Log.logtime.desc())

        logs = query.all()

        return render_template('log_query.html', logs=logs, keyword=keyword, logtypes=logtypes, order=order)

    return redirect(url_for('login'))


@app.route('/log_detail/<int:logid>')
def log_detail(logid):
    log = Log.query.get_or_404(logid)
    return render_template('log_detail.html', log=log)


if __name__ == '__main__':
    app.run(debug=True)
