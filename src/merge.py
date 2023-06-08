import os, sys, logging, argparse, json, warnings, pandas, datetime
from IPython.display import Image, display 
warnings.filterwarnings(action='ignore')
import numpy as np
from dateutil.relativedelta import relativedelta
from pykrx import stock, bond
import pandas_datareader.data as web

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.style.use('seaborn-whitegrid')

from src import settings, utils, data_manager

def rltrader(__epoch, __mode, __ver, __name, __stock_code, __rl_method, __net, __backend,
             __start_date, __end_date, __lr, __discount_factor, __balance) :
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'update', 'predict'], default=__mode)
    parser.add_argument('--ver', choices=['v1', 'v2', 'v3', 'v4', 'v4.1', 'v4.2'], default=__ver)
    parser.add_argument('--name', default=__name)
    parser.add_argument('--stock_code', nargs='+', default=[__stock_code])
    parser.add_argument('--rl_method', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'monkey'], default=__rl_method)
    parser.add_argument('--net', choices=['dnn', 'lstm', 'cnn', 'monkey'], default=__net)
    parser.add_argument('--backend', choices=['pytorch', 'tensorflow', 'plaidml'], default=__backend)
    parser.add_argument('--start_date', default=__start_date)
    parser.add_argument('--end_date', default=__end_date)
    parser.add_argument('--lr', type=float, default=__lr)
    parser.add_argument('--discount_factor', type=float, default=__discount_factor)
    parser.add_argument('--balance', type=int, default=__balance)
    parser.add_argument('--epoch', type=int, default=__epoch)
    args = parser.parse_args(args=[])
    print('- Parameters of test case :\n', args)
    
    # 학습기 파라미터 설정
    output_name = f'{args.mode}_{args.name}_{args.rl_method}_{args.net}'
    learning = args.mode in ['train', 'update']
    reuse_models = args.mode in ['test', 'update', 'predict']
    value_network_name = f'{args.name}_{args.rl_method}_{args.net}_value.mdl'
    policy_network_name = f'{args.name}_{args.rl_method}_{args.net}_policy.mdl'
    start_epsilon = 1 if args.mode in ['train', 'update'] else 0
    num_epoches = args.epoch if args.mode in ['train', 'update'] else 1
    num_steps = 5 if args.net in ['lstm', 'cnn'] else 1
    
    # Backend 설정
    os.environ['RLTRADER_BACKEND'] = args.backend
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
    
    # 출력 경로 생성
    output_path = os.path.join(settings.BASE_DIR, 'output', output_name)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
        
    # 파라미터 기록
    params = json.dumps(vars(args))
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(params)
    
    # 모델 경로 준비
    # 모델 포멧은 TensorFlow는 h5, PyTorch는 pickle
    value_network_path = os.path.join(settings.BASE_DIR, 'models', value_network_name)
    policy_network_path = os.path.join(settings.BASE_DIR, 'models', policy_network_name)

    # 로그 기록 설정
    log_path = os.path.join(output_path, f'{output_name}.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger(settings.LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    #logger.info(params)

    # Backend 설정, 로그 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from src.learners import ReinforcementLearner, DQNLearner, \
        PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_price = []
    list_max_trading_price = []

    for stock_code in args.stock_code:
        # 차트 데이터, 학습 데이터 준비
        chart_data, training_data = data_manager.load_data(
            stock_code, args.start_date, args.end_date, ver=args.ver)

        assert len(chart_data) >= num_steps

        # 최소/최대 단일 매매 금액 설정
        min_trading_price = 100000
        max_trading_price = 10000000

        # 공통 파라미터 설정
        common_params = {'rl_method': args.rl_method, 
            'net': args.net, 'num_steps': num_steps, 'lr': args.lr,
            'balance': args.balance, 'num_epoches': num_epoches, 
            'discount_factor': args.discount_factor, 'start_epsilon': start_epsilon,
            'output_path': output_path, 'reuse_models': reuse_models}

        # 강화학습 시작
        learner = None
        if args.rl_method != 'a3c':
            common_params.update({'stock_code': stock_code,
                'chart_data': chart_data, 
                'training_data': training_data,
                'min_trading_price': min_trading_price, 
                'max_trading_price': max_trading_price})
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params, 
                    'value_network_path': value_network_path})
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'a2c':
                learner = A2CLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'monkey':
                common_params['net'] = args.rl_method
                common_params['num_epoches'] = 10
                common_params['start_epsilon'] = 1
                learning = False
                learner = ReinforcementLearner(**common_params)
        else:
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_price.append(min_trading_price)
            list_max_trading_price.append(max_trading_price)

    if args.rl_method == 'a3c':
        learner = A3CLearner(**{
            **common_params, 
            'list_stock_code': list_stock_code, 
            'list_chart_data': list_chart_data, 
            'list_training_data': list_training_data,
            'list_min_trading_price': list_min_trading_price, 
            'list_max_trading_price': list_max_trading_price,
            'value_network_path': value_network_path, 
            'policy_network_path': policy_network_path})
        
    assert learner is not None

    if args.mode in ['train', 'test', 'update']:
        learner.run(learning=learning)
        if args.mode in ['train', 'update']:
            learner.save_models()
    elif args.mode == 'predict':
        learner.predict()
    
    return args

def calc_mdd(list_x, list_pv):

    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    years_fmt = mdates.DateFormatter('%Y')
  
    list_x = [datetime.datetime.strptime(date, '%Y%m%d').date() for date in list_x]
    arr_pv = np.array(list_pv)
    peak_lower = np.argmax(np.maximum.accumulate(arr_pv) - arr_pv)
    peak_upper = np.argmax(arr_pv[:peak_lower])
    
    idx_min = np.argmin(arr_pv)
    idx_max = np.argmax(arr_pv)

    _, ax = plt.subplots()
    ax.plot(list_x, arr_pv, color='gray')
    ax.plot([list_x[peak_upper], list_x[peak_lower]], [arr_pv[peak_upper], arr_pv[peak_lower]], '-', color='blue')
    ax.plot([list_x[idx_min]], [arr_pv[idx_min]], 'v', color='blue')
    ax.plot([list_x[idx_max]], [arr_pv[idx_max]], '^', color='red')
    
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    
    plt.show()

    return (arr_pv[peak_lower] - arr_pv[peak_upper]) / arr_pv[peak_upper]

def preprocess(data):
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        data['close_ma{}'.format(window)] = \
            data['close'].rolling(window).mean()
        data['volume_ma{}'.format(window)] = \
            data['volume'].rolling(window).mean()
        data['close_ma%d_ratio' % window] = \
            (data['close'] - data['close_ma%d' % window]) \
            / data['close_ma%d' % window]
        data['volume_ma%d_ratio' % window] = \
            (data['volume'] - data['volume_ma%d' % window]) \
            / data['volume_ma%d' % window]

    data['open_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'open_lastclose_ratio'] = \
        (data['open'][1:].values - data['close'][:-1].values) \
        / data['close'][:-1].values
    data['high_close_ratio'] = \
        (data['high'].values - data['close'].values) \
        / data['close'].values
    data['low_close_ratio'] = \
        (data['low'].values - data['close'].values) \
        / data['close'].values
    data['close_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'close_lastclose_ratio'] = \
        (data['close'][1:].values - data['close'][:-1].values) \
        / data['close'][:-1].values
    data['volume_lastvolume_ratio'] = np.zeros(len(data))
    data.loc[1:, 'volume_lastvolume_ratio'] = \
        (data['volume'][1:].values - data['volume'][:-1].values) \
        / data['volume'][:-1] \
            .replace(to_replace=0, method='ffill') \
            .replace(to_replace=0, method='bfill').values

    for window in windows:
        data[f'inst_ma{window}'] = data['close'].rolling(window).mean()
        #data[f'frgn_ma{window}'] = data['volume'].rolling(window).mean()
        data[f'inst_ma{window}_ratio'] = \
            (data['close'] - data[f'inst_ma{window}']) / data[f'inst_ma{window}']
        #data[f'frgn_ma{window}_ratio'] = \
        #    (data['volume'] - data[f'frgn_ma{window}']) / data[f'frgn_ma{window}']
        data['inst_lastinst_ratio'] = np.zeros(len(data))
        data.loc[1:, 'inst_lastinst_ratio'] = (
            (data['inst'][1:].values - data['inst'][:-1].values)
            / data['inst'][:-1].replace(to_replace=0, method='ffill')\
                .replace(to_replace=0, method='bfill').values
        )
        data[f'ind_ma{window}'] = data['ind'].rolling(window).mean()
        data[f'foreign_ma{window}'] = data['foreign'].rolling(window).mean()

    return data

def Collecting_new_data(Stock_code,train_start_date,train_end_date,test_start_date,test_end_date):
    datetime_object = datetime.datetime.strptime(train_start_date, '%Y-%m-%d')
    train_start_date2=(datetime_object - relativedelta(months=7)).strftime('%Y-%m-%d')
    
    df = web.DataReader(Stock_code, 'naver',
      start=train_start_date2, end=test_end_date)
    
    df.index.name, df.columns = 'date', ['open','high','low','close','volume']
    df = df.astype(float)

    df_part1 = stock.get_market_fundamental(train_start_date2.replace('-', ''), test_end_date.replace('-', ''), Stock_code)
    df['per']=df_part1['PER']
    df['pbr']=df_part1['PBR']
    df['roe']=df_part1['PBR']/df_part1['PER']*100
    df['diffratio']=df['close'].diff()*100

    df_part2 = stock.get_market_trading_value_by_date(train_start_date2.replace('-', ''), test_end_date.replace('-', ''), Stock_code)
    df['ind']=df_part2['개인']
    df['ind_diff']=df_part2['개인'].diff()
    df['inst']=df_part2['기관합계']
    df['inst_diff']=df_part2['기관합계'].diff()
    df['foreign']=df_part2['외국인합계']
    df['foreign_diff']=df_part2['외국인합계'].diff()

    df = preprocess(df)
    df2 = df.dropna()

    df2.to_csv(f'/content/drive/MyDrive/rl_trader_workspace/src/data/v3/{Stock_code}.csv')

    return df2